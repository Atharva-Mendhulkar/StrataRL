import torch
import numpy as np
from typing import List, Dict

LENGTH_NORM_CLAMP = 256   # completions longer than this get equal treatment
ADVANTAGE_CLIP    = 5.0


def compute_san_advantages(
    rewards:    torch.Tensor,   # [B, G] — GDPO-aggregated composite rewards
    domains:    List[str],      # length B — domain label per prompt
    eps:        float = 1e-8,
) -> torch.Tensor:
    """
    Stratified Advantage Normalization.

    Computes Z-score normalization WITHIN each domain stratum independently.
    Prevents cross-stratum bias from heterogeneous reward distributions.

    Returns advantages: [B, G], clipped to ±ADVANTAGE_CLIP
    """
    advantages = torch.zeros_like(rewards)
    unique_domains = set(domains)

    for domain in unique_domains:
        stratum_idx = [i for i, d in enumerate(domains) if d == domain]

        if len(stratum_idx) < 2:
            # Single-sample stratum: zero advantage (no comparison possible)
            advantages[stratum_idx] = 0.0
            continue

        stratum_rewards = rewards[stratum_idx]       # [n_d, G]
        
        # ── Reward Consistency Guard ──────────────────────────────────────────
        # If signal is weak, fallback to centered rewards instead of zeroing out
        # This preserves directionality while avoiding noise amplification
        if stratum_rewards.std() < 1e-3:
            centered = stratum_rewards - stratum_rewards.mean()
            advantages[stratum_idx] = torch.clamp(centered, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)
            continue

        flat  = stratum_rewards.flatten()
        mu    = flat.mean()
        sigma = flat.std() + eps

        normalized               = (stratum_rewards - mu) / sigma
        advantages[stratum_idx]  = torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    return advantages


def expand_advantages_to_tokens(
    advantages:          torch.Tensor,        # [B, G]
    completion_lengths:  List[List[int]],     # [B][G] — token counts per completion
    use_length_norm:     bool = True,
) -> torch.Tensor:
    """
    Expand scalar per-completion advantages to per-token advantages.
    Applies linear length normalization to prevent excessive gradient shrinkage.
    """
    token_advantages = []
    for b in range(advantages.shape[0]):
        for g in range(advantages.shape[1]):
            adv_scalar = advantages[b, g].item()
            length     = completion_lengths[b][g]

            if use_length_norm and length > 0:
                clamped_length = min(length, LENGTH_NORM_CLAMP)
                norm_factor    = 1.0 / (1.0 + 0.01 * clamped_length)
                adv_scalar     = adv_scalar * norm_factor

            token_advantages.extend([adv_scalar] * length)

    return torch.tensor(token_advantages, dtype=torch.float32)
