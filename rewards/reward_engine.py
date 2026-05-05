import torch
from typing import List, Dict
from rewards.structural_reward import structural_reward
from rewards.token_repetition  import token_repetition_penalty
from rewards.outcome_verifiers import DOMAIN_VERIFIERS


def gdpo_normalize(
    x: torch.Tensor, 
    threshold: float = 1e-3,
    cooldown_active: bool = False
) -> torch.Tensor:
    """
    Z-normalize a [B, G] reward tensor across G per row.
    If std < threshold and cooldown is not active, inject randomized structured variance.
    """
    mu  = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    B, G = x.shape
    
    # Standard normalization
    z = (x - mu) / (std + 1e-8)
    
    # Prevent zero-variance -> zero-gradient collapse
    # If cooldown is active, we skip noise injection to allow the policy to stabilize
    low_var_mask = (std < threshold) & (~cooldown_active)
    
    if low_var_mask.any():
        mask_sq = low_var_mask.squeeze(-1)
        noise_base = torch.linspace(-0.01, 0.01, steps=G, device=x.device)
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
    gdpo_cooldown: bool = False, # Pass cooldown state from training loop
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score a batch of rollouts with support for GDPO stabilization cooldowns.
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

    struct_gated = struct_r * token_rep

    # GDPO normalization with cooldown-aware noise injection
    z_outcome = gdpo_normalize(outcome_r, cooldown_active=gdpo_cooldown)
    z_struct  = gdpo_normalize(struct_gated, cooldown_active=gdpo_cooldown)

    gdpo_rewards = w_outcome_eff * z_outcome + w_struct_eff * z_struct
    raw_rewards  = torch.stack([outcome_r, struct_r, token_rep], dim=0)

    return gdpo_rewards, raw_rewards
