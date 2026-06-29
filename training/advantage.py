# training/advantage.py

import torch
from typing import List

LENGTH_NORM_CLAMP     = 512
ADVANTAGE_CLIP        = 5.0
SAN_ZERO_VAR_THRESH   = 1e-2
SAN_LOW_VAR_THRESH    = 0.05


def compute_global_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Standard GRPO global advantage normalization (Condition B).
    Computes mean and std across the entire batch, ignoring domain strata.
    Applies the same clipping and zero-variance rules as SAN for a fair ablation.
    """
    flat = rewards.flatten()
    mu = flat.mean()
    sigma = flat.std()

    if sigma < SAN_ZERO_VAR_THRESH:
        centered = rewards - mu
        return torch.clamp(centered, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)
    
    if sigma < SAN_LOW_VAR_THRESH:
        damping_factor = sigma.item() / SAN_LOW_VAR_THRESH
        normalized = (rewards - mu) / (sigma + eps)
        dampened = normalized * damping_factor
        return torch.clamp(dampened, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    normalized = (rewards - mu) / (sigma + eps)
    return torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)



def compute_san_advantages(
    rewards:  torch.Tensor,   # [B, G] — GDPO-normalized composite rewards
    domains:  List[str],
    eps:      float = 1e-8,
) -> torch.Tensor:
    """
    Stratified Advantage Normalization. Unchanged from v2.0.

    std < SAN_ZERO_VAR_THRESH (1e-2):   center-without-scale
    SAN_ZERO_VAR_THRESH <= std < 0.05:  partial-credit dampening
    std >= SAN_LOW_VAR_THRESH (0.05):   full Z-normalization
    """
    advantages     = torch.zeros_like(rewards)
    unique_domains = set(domains)

    for domain in unique_domains:
        stratum_idx = [i for i, d in enumerate(domains) if d == domain]

        if len(stratum_idx) < 2:
            advantages[stratum_idx] = 0.0
            continue

        stratum_rewards = rewards[stratum_idx]
        flat            = stratum_rewards.flatten()
        mu              = flat.mean()
        sigma           = flat.std()

        if sigma < SAN_ZERO_VAR_THRESH:
            centered                = stratum_rewards - mu
            advantages[stratum_idx] = torch.clamp(centered, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        elif sigma < SAN_LOW_VAR_THRESH:
            damping_factor          = sigma.item() / SAN_LOW_VAR_THRESH
            normalized               = (stratum_rewards - mu) / (sigma + eps)
            dampened                 = normalized * damping_factor
            advantages[stratum_idx] = torch.clamp(dampened, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        else:
            normalized               = (stratum_rewards - mu) / (sigma + eps)
            advantages[stratum_idx] = torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    return advantages


# ─────────────────────────────────────────────────────────────────────────────
# PATCH I-10 — REGRESSION FIX (EXP_01 step 500: avg_think_len collapsed to ~30-40
# tokens, GSM8K/MMLU/StrategyQA all regressed below the measured baseline)
#
# ROOT CAUSE: 1/sqrt(L) per-token normalization gave SHORT completions a LARGER
# per-token advantage magnitude than LONG completions for the SAME raw
# advantage:
#     norm_factor(L=30)  = 1/sqrt(30)  ≈ 0.183
#     norm_factor(L=200) = 1/sqrt(200) ≈ 0.071    (2.6x weaker per token)
#
# This bias was LATENT until combined with the structural_reward tag-presence
# loophole (see rewards/structural_reward.py PATCH I-11): a 30-40 token
# completion like <decompose>5*17</decompose><compute>85</compute>
# <verify>85</verify> passed structural_reward=1.0 with near-zero reasoning
# content, so A_short ≈ A_long for "correct" completions. With equal raw
# advantage, 1/sqrt(L) gave short completions a ~2.6x stronger per-token
# push — a direct, compounding incentive toward brevity over 500 steps.
#
# FIX: every token in a completion now receives the SAME scalar advantage
# (no length division). Combined with I-11 (which ensures A_long > A_short
# by requiring genuine per-tag content), the natural advantage gap between
# long-correct and short-lucky completions now survives unmodified into the
# per-token gradient instead of being diluted by sqrt(L) ratios.
#
# Verbose-faking / reasoning-loop risk (the ORIGINAL motivation for length
# norm) is covered by orthogonal mechanisms that don't reintroduce a
# length-based gradient bias:
#   - token_repetition_penalty (rewards/token_repetition.py)
#   - GARBAGE_PATTERNS in structural_reward
#   - hard max_new_tokens cap at the rollout level
#   - I-11's per-tag content checks (padding inside a tag still has to be
#     non-repetitive AND clear the per-tag minimum, it just can't be empty)
#
# use_length_norm defaults to False. The sqrt(clamped_L) path is retained
# ONLY for ablation studies (e.g. "what if we re-enable length norm" as a
# control condition) — do not enable in production without re-running the
# 50-step M4 smoke test AND confirming I-11 is active.
# ─────────────────────────────────────────────────────────────────────────────

def expand_advantages_to_tokens(
    advantages:          torch.Tensor,        # [B, G]
    completion_lengths:  List[List[int]],     # [B][G] — token counts per completion
    use_length_norm:     bool = False,         # PATCH I-10: was True, now False
) -> torch.Tensor:
    """
    Expand scalar per-completion advantages to per-token advantages.

    Default (use_length_norm=False): every token in a completion receives
    the completion's full scalar advantage, unmodified. A completion's
    contribution to the batch-averaged loss scales as A * L — long, correct
    completions now dominate the gradient over short, lucky ones, which is
    the desired direction once I-11 ensures A_long > A_short.

    use_length_norm=True (ablation only): retains the pre-patch 1/sqrt(L)
    behavior that produced the step-500 think-collapse. Do not use in
    production.
    """
    token_advantages = []
    for b in range(advantages.shape[0]):
        for g in range(advantages.shape[1]):
            adv_scalar = advantages[b, g].item()
            length     = completion_lengths[b][g]

            if use_length_norm and length > 0:
                clamped_length = min(length, LENGTH_NORM_CLAMP)
                norm_factor    = 1.0 / (clamped_length ** 0.5)
                adv_scalar     = adv_scalar * norm_factor

            token_advantages.extend([adv_scalar] * length)

    return torch.tensor(token_advantages, dtype=torch.float32)
