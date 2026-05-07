import torch
from typing import List, Dict
from rewards.structural_reward import structural_reward
from rewards.token_repetition  import token_repetition_penalty
from rewards.outcome_verifiers import DOMAIN_VERIFIERS


REWARD_CLIP_RANGE    = 2.0
GDPO_NOISE_FRACTION  = 0.005        # noise = ±(fraction × clip_span)
GDPO_CLIP_SPAN       = REWARD_CLIP_RANGE * 2   # = 4.0
GDPO_ZERO_VAR_THRESH = 1e-2         # RAISED from 1e-3: partial credit has real std


def get_gdpo_noise_magnitude(step: int) -> float:
    anneal_steps = 200
    noise_start  = GDPO_NOISE_FRACTION   # 0.005
    noise_floor  = 0.001
    if step >= anneal_steps:
        fraction = noise_floor
    else:
        # Linear annealing from start to floor
        fraction = noise_start - (step / anneal_steps) * (noise_start - noise_floor)
    return fraction * GDPO_CLIP_SPAN


def gdpo_normalize(x: torch.Tensor, step: int = 0) -> torch.Tensor:
    """
    For rows with std >= GDPO_ZERO_VAR_THRESH: standard Z-normalization.
    For rows with std < GDPO_ZERO_VAR_THRESH: inject annealed synthetic ranking noise.
    Noise magnitude = GDPO_NOISE_FRACTION × GDPO_CLIP_SPAN, annealed over 200 steps.
    """
    B, G = x.shape
    z    = torch.zeros_like(x)
    eta  = get_gdpo_noise_magnitude(step)

    for i in range(B):
        row_std = x[i].std().item()
        if row_std >= GDPO_ZERO_VAR_THRESH:
            mu   = x[i].mean()
            sig  = x[i].std() + 1e-8
            z[i] = (x[i] - mu) / sig
        else:
            noise = torch.linspace(-eta, eta, steps=G, device=x.device)
            perm  = torch.randperm(G, device=x.device)
            z[i]  = noise[perm]
    return z


def clip_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """Clip BEFORE GDPO and SAN. This is the mandated operation order."""
    return torch.clamp(rewards, -REWARD_CLIP_RANGE, REWARD_CLIP_RANGE)


def score_batch(
    rollouts:      List[Dict],
    ground_truths: List[str],
    domains:       List[str],
    w_outcome:     float = 0.7,
    w_struct:      float = 0.3,
    phase:         str = "strict",
    cooldown:      bool = False,
    step:          int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    I-1: score_batch with reward clipping BEFORE normalization.
    """
    B = len(rollouts)
    G = len(rollouts[0]["completions"])

    outcome_r = torch.zeros(B, G)
    struct_r  = torch.zeros(B, G)
    token_rep = torch.ones(B, G)

    for i, (rollout, gt, domain) in enumerate(zip(rollouts, ground_truths, domains)):
        verifier = DOMAIN_VERIFIERS.get(domain, DOMAIN_VERIFIERS["gsm8k"])
        for j, completion in enumerate(rollout["completions"]):
            outcome_r[i, j] = verifier(completion, gt)
            struct_r[i, j]  = structural_reward(completion, domain=domain, phase=phase)
            token_rep[i, j] = token_repetition_penalty(rollout["token_ids"][j])

    struct_gated = struct_r * token_rep
    
    # I-1: Reward Clipping BEFORE normalization
    outcome_clipped = clip_rewards(outcome_r)
    struct_clipped  = clip_rewards(struct_gated)

    # I-1: GDPO normalization with step-aware annealing
    z_outcome = gdpo_normalize(outcome_clipped, step=step)
    z_struct  = gdpo_normalize(struct_clipped, step=step)

    combined_rewards = w_outcome * z_outcome + w_struct * z_struct
    raw_rewards      = torch.stack([outcome_r, struct_r, token_rep], dim=0)

    return combined_rewards, raw_rewards
