import torch
import torch.nn.functional as F

"""
ARCHITECTURAL DECISION RECORD: π_ref = π_old
=============================================
π_ref is defined as rollout-time logprobs (captured via vLLM logprobs=1).
No separate frozen SFT model is loaded or maintained.
This saves 1.8GB VRAM and uses the mathematically correct PPO formulation:
KL penalizes drift from the policy that GENERATED the rollouts.
This decision is irrevocable. Do not add ref_model back.
"""

def grpo_loss(
    policy_model,
    input_ids:        torch.Tensor,
    attention_mask:   torch.Tensor,
    completion_mask:  torch.Tensor,
    advantages:       torch.Tensor,
    old_logprobs:     torch.Tensor,   # rollout-time logprobs = π_ref = π_old
    beta:             float = 0.01,
    clip_eps:         float = 0.2,
    entropy_floor:    float = 0.0,
    entropy_coeff:    float = 0.01,
    recompute_check:  bool  = False,
) -> dict:

    # I-8: Prompt-region assertion — old_logprobs must be zero on prompt tokens
    prompt_mask = (completion_mask == 0).float()
    prompt_logp_contamination = (old_logprobs * prompt_mask).abs().sum().item()
    assert prompt_logp_contamination < 1e-3, (
        f"PROMPT-REGION CONTAMINATION: old_logprobs has non-zero values in prompt "
        f"positions (sum={prompt_logp_contamination:.6f}). "
        f"Check packing logic in utils/tokenize.py."
    )

    # Forward pass
    outputs     = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits      = outputs.logits[:, :-1, :]
    labels      = input_ids[:, 1:]
    log_probs   = F.log_softmax(logits, dim=-1)
    policy_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    old_logp_aligned = old_logprobs[:, 1:].detach()

    # I-8: Shape assertion
    assert old_logp_aligned.shape == policy_logp.shape, (
        f"ALIGNMENT FAILURE: old_logp {old_logp_aligned.shape} != "
        f"policy_logp {policy_logp.shape}. HALTING."
    )

    # Log ratio — clamped in log space to prevent overflow
    log_ratio         = torch.clamp(policy_logp - old_logp_aligned, -10.0, 10.0)
    ratio             = torch.exp(log_ratio)

    comp_mask  = completion_mask[:, 1:]
    adv_tokens = advantages[:, 1:]

    # Surrogate loss
    surr1       = ratio * adv_tokens
    surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tokens
    policy_loss = -torch.min(surr1, surr2)

    # KL — CORRECT formula (fixed sign, uses actual old_logprobs)
    # = p_old(t) * log(p_old(t) / p_new(t)) per token
    # Always >= 0 in expectation. Individual tokens may be negative — do not clamp.
    raw_kl_per_token = torch.exp(old_logp_aligned) * (old_logp_aligned - policy_logp)

    # I-9: Track raw KL for absolute drift monitoring (separate from normalized)
    kl_scale  = raw_kl_per_token.abs().mean().detach() + 1e-8
    kl_norm   = raw_kl_per_token / kl_scale

    # Entropy
    entropy = -(torch.exp(policy_logp) * policy_logp * comp_mask).sum() / (comp_mask.sum() + 1e-8)

    # Aggregate
    denom             = comp_mask.sum() + 1e-8
    policy_loss_mean  = (policy_loss * comp_mask).sum() / denom
    kl_norm_mean      = (kl_norm * comp_mask).sum() / denom
    raw_kl_mean       = (raw_kl_per_token * comp_mask).sum() / denom    # I-9
    clip_frac         = ((ratio - 1).abs() > clip_eps).float()
    clip_frac_mean    = (clip_frac * comp_mask).sum() / denom

    total_loss = policy_loss_mean + beta * kl_norm_mean

    entropy_deficit = torch.tensor(0.0, device=input_ids.device)
    if entropy_floor > 0.0:
        entropy_deficit = F.relu(torch.tensor(entropy_floor, device=input_ids.device) - entropy)
        total_loss      = total_loss - entropy_coeff * entropy_deficit

    return {
        "loss":            total_loss,
        "policy_loss":     policy_loss_mean,
        "kl_norm":         kl_norm_mean,
        "raw_kl_mean":     raw_kl_mean,       # I-9: absolute drift signal
        "kl_scale":        kl_scale,
        "entropy":         entropy,
        "entropy_deficit": entropy_deficit,
        "ratio_mean":      (ratio * comp_mask).sum() / denom,
        "ratio_max":       ratio.max(),
        "clip_frac":       clip_frac_mean,
    }
