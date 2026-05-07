# training/advantage.py

import torch
from typing import List

LENGTH_NORM_CLAMP   = 512
ADVANTAGE_CLIP      = 5.0
SAN_ZERO_VAR_THRESH = 1e-2    # raised from 1e-3
SAN_LOW_VAR_THRESH  = 0.05    # new: dampening threshold for weak partial credit


def compute_san_advantages(
    rewards: torch.Tensor,   # [B, G]
    domains: List[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Three normalization regimes:
    std < 1e-2:        center-without-scale (true zero variance)
    1e-2 <= std < 0.05: dampen to prevent ghost amplification of partial credit
    std >= 0.05:       full Z-normalization
    """
    advantages    = torch.zeros_like(rewards)
    unique_domains = set(domains)

    for domain in unique_domains:
        idx = [i for i, d in enumerate(domains) if d == domain]
        if len(idx) < 2:
            advantages[idx] = 0.0
            continue

        stratum = rewards[idx]
        flat    = stratum.flatten()
        mu      = flat.mean()
        sigma   = flat.std()

        if sigma < SAN_ZERO_VAR_THRESH:
            centered         = stratum - mu
            advantages[idx]  = torch.clamp(centered, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        elif sigma < SAN_LOW_VAR_THRESH:
            damping          = sigma.item() / SAN_LOW_VAR_THRESH
            normalized       = (stratum - mu) / (sigma + eps)
            advantages[idx]  = torch.clamp(normalized * damping, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        else:
            normalized       = (stratum - mu) / (sigma + eps)
            advantages[idx]  = torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    return advantages


def expand_advantages_to_tokens(
    advantages: torch.Tensor,
    completion_lengths: List[List[int]],
    use_length_norm: bool = True,
) -> torch.Tensor:
    token_advantages = []
    for b in range(advantages.shape[0]):
        for g in range(advantages.shape[1]):
            adv    = advantages[b, g].item()
            length = completion_lengths[b][g]
            if use_length_norm and length > 0:
                clamped = min(length, LENGTH_NORM_CLAMP)
                adv     = adv / (clamped ** 0.5)
            token_advantages.extend([adv] * length)
    return torch.tensor(token_advantages, dtype=torch.float32)
