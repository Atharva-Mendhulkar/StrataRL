import torch
import torch.nn.functional as F

def grpo_loss(
    policy_model:      torch.nn.Module,
    input_ids:         torch.Tensor,    # [B*G, seq_len]
    attention_mask:    torch.Tensor,    # [B*G, seq_len]
    completion_mask:   torch.Tensor,    # [B*G, seq_len] — 1 for completion tokens
    advantages:        torch.Tensor,    # [B*G, seq_len] — expanded token advantages
    old_logprobs:      torch.Tensor,    # [B*G, seq_len] — logprobs from rollout
    beta:              float = 0.01,    # KL penalty weight
    clip_eps:          float = 0.2,     # PPO clipping
) -> dict:
    """
    Computes GRPO loss with stabilized ratio and normalized KL penalty.
    """
    # ── Forward pass ─────────────────────────────────────────────────────────
    outputs = policy_model(input_ids, attention_mask=attention_mask)
    logits  = outputs.logits
    
    # Shift labels for causal LM objective
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Extract logprobs for actual tokens [B*G, seq_len-1]
    policy_logp = torch.gather(
        log_probs[:, :-1, :], 
        dim=-1, 
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    # Align masks and old_logprobs to the shifted policy_logp
    completion_mask_aligned = completion_mask[:, 1:]
    old_logp_aligned        = old_logprobs[:, 1:]
    advantages_aligned      = advantages[:, 1:]

    # ── Surrogate Loss (PPO-style) ──────────────────────────────────────────
    # ratio = exp(policy_logp - old_logp)
    log_ratio = policy_logp - old_logp_aligned
    clamped_log_ratio = torch.clamp(log_ratio, -10, 10)
    ratio     = torch.exp(clamped_log_ratio)
    
    ratio_clipped_frac = (clamped_log_ratio.abs() >= 9.5).float().mean()
    
    surr1 = ratio * advantages_aligned
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_aligned
    policy_loss = -torch.min(surr1, surr2)

    # ── KL Penalty (D_KL(old || new)) ───────────────────────────────────────
    raw_kl = torch.exp(old_logp_aligned) * (old_logp_aligned - policy_logp)
    
    # KL Normalization: stabilize beta tuning without unstable gradient scaling
    kl_scale = raw_kl.abs().mean().detach() + 1e-8
    kl_per_token_norm = raw_kl / kl_scale
    
    # ── Total Loss ──────────────────────────────────────────────────────────
    total_loss_per_token = policy_loss + beta * kl_per_token_norm
    
    masked_loss = (total_loss_per_token * completion_mask_aligned).sum()
    norm_factor = completion_mask_aligned.sum() + 1e-8
    
    # Diagnostics
    with torch.no_grad():
        entropy = -(torch.exp(policy_logp) * policy_logp * completion_mask_aligned).sum() / norm_factor
        kl_val  = (raw_kl * completion_mask_aligned).sum() / norm_factor
        raw_kl_mean = kl_val # Alias for clarity

    return {
        "loss":               masked_loss / norm_factor,
        "policy_loss":        (policy_loss * completion_mask_aligned).sum() / norm_factor,
        "kl":                 kl_val,
        "raw_kl_mean":        raw_kl_mean,
        "entropy":            entropy,
        "ratio_clipped_frac": ratio_clipped_frac,
    }
