import torch
from typing import List, Dict
from rewards.structural_reward import structural_reward
from rewards.token_repetition  import token_repetition_penalty
from rewards.outcome_verifiers import DOMAIN_VERIFIERS


def gdpo_normalize(x: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """
    Z-normalize a [B, G] reward tensor across G per row.
    If std < threshold, inject randomized structured variance to preserve gradient direction.
    """
    mu  = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    B, G = x.shape
    
    # Standard normalization
    z = (x - mu) / (std + 1e-8)
    
    # Prevent zero-variance -> zero-gradient collapse
    low_var_mask = std < threshold
    
    # For low variance groups, inject minimal randomized structured variance
    if low_var_mask.any():
        mask_sq = low_var_mask.squeeze(-1)
        # Create base noise
        noise_base = torch.linspace(-0.01, 0.01, steps=G, device=x.device)
        # Apply to each low-variance row with a distinct permutation
        # This guarantees per-row independence and zero-mean preservation
        for i in range(B):
            if mask_sq[i]:
                perm = torch.randperm(G, device=x.device)
                z[i] = noise_base[perm]
    
    return z


def score_batch(
    rollouts:      List[Dict],
    ground_truths: List[str],
    domains:       List[str],
    w_outcome:     float = 0.7,
    w_struct:      float = 0.3,
    phase:         str = "strict",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score a batch of rollouts.
    
    In 'bootstrap' phase:
    - structural weight is increased to 0.7 to provide stronger formatting signal
    
    Returns:
        gdpo_rewards: [B, G]  — GDPO-aggregated composite reward
        raw_rewards:  [3, B, G] — [outcome, struct, token_rep] before aggregation
    """
    if phase == "bootstrap":
        w_outcome_eff = 0.3
        w_struct_eff  = 0.7
    else:
        w_outcome_eff = w_outcome
        w_struct_eff  = w_struct

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
            token_rep[i, j] = token_repetition_penalty(
                rollout["token_ids"][j]
            )

    # Gate structural with token repetition
    struct_gated = struct_r * token_rep

    # GDPO: normalize each signal independently, then combine
    z_outcome = gdpo_normalize(outcome_r)
    z_struct  = gdpo_normalize(struct_gated)

    gdpo_rewards = w_outcome_eff * z_outcome + w_struct_eff * z_struct
    raw_rewards  = torch.stack([outcome_r, struct_r, token_rep], dim=0)

    return gdpo_rewards, raw_rewards
