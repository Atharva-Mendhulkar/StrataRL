import torch
import wandb
import yaml
import random
from collections import defaultdict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from m4.m4_rollout_engine      import M4RolloutEngine
from rewards.reward_engine      import score_batch
from training.advantage         import compute_san_advantages, expand_advantages_to_tokens
from training.policy_update     import grpo_loss
from curriculum.ucb_scheduler   import UCBCurriculumScheduler
from data.loaders               import load_domain_samples
from monitoring.monitor         import SmokeTestMonitor


def _sample_batch(samples, batch_size):
    return random.sample(samples, min(batch_size, len(samples)))


def _pack_rollouts(rollouts, tokenizer, device):
    """
    Pack rollouts into tensors for policy update.
    Returns: input_ids, attention_mask, completion_mask, old_logprobs
    """
    all_input_ids = []
    all_att_masks = []
    all_comp_masks = []
    all_old_lps = []
    
    max_len = 0
    # First pass to find max length
    for r in rollouts:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=True).input_ids
        for comp_ids in r["token_ids"]:
            max_len = max(max_len, len(prompt_ids) + len(comp_ids))

    for r in rollouts:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=True).input_ids
        for i, comp_ids in enumerate(r["token_ids"]):
            full_ids = prompt_ids + comp_ids
            pad_len = max_len - len(full_ids)
            
            input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            att_mask  = [1] * len(full_ids) + [0] * pad_len
            comp_mask = [0] * len(prompt_ids) + [1] * len(comp_ids) + [0] * pad_len
            
            # Logprobs need padding too
            old_lps = [0.0] * len(prompt_ids) + r["rollout_logprobs"][i] + [0.0] * pad_len
            
            all_input_ids.append(input_ids)
            all_att_masks.append(att_mask)
            all_comp_masks.append(comp_mask)
            all_old_lps.append(old_lps)

    input_ids_t    = torch.tensor(all_input_ids, device=device)
    attention_mask_t = torch.tensor(all_att_masks, device=device)
    completion_mask_t = torch.tensor(all_comp_masks, device=device)
    old_logprobs_t = torch.tensor(all_old_lps, device=device)

    # ── Packing Alignment Guard ───────────────────────────────────────────────
    assert old_logprobs_t.shape == input_ids_t.shape, \
        f"Packing Error: logprobs {old_logprobs_t.shape} != input_ids {input_ids_t.shape}"

    return input_ids_t, attention_mask_t, completion_mask_t, old_logprobs_t


def _expand_to_seq(token_advs, shape, completion_mask):
    """Expand flat token advantages to [B*G, seq_len] tensor aligned with completion_mask."""
    res = torch.zeros(shape, device=completion_mask.device)
    res[completion_mask == 1] = token_advs.to(completion_mask.device)
    return res


def run_smoke_test(config_path: str = "m4/m4_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device    = cfg["device"] if torch.backends.mps.is_available() else "cpu"
    print(f"[StrataRL M4] Device: {device}")

    # ── Model setup ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = cfg["lora_r"],
        lora_alpha     = cfg["lora_alpha"],
        target_modules = cfg["target_modules"],
        bias           = "none",
    )
    model     = get_peft_model(base_model, lora_config)
    
    # ── Final Pre-Run Sanity Snapshot ─────────────────────────────────────────
    print({
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "G": cfg["G"],
        "beta": cfg["beta"],
        "temperature": cfg["temperature"]
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    rollout_engine = M4RolloutEngine(model, tokenizer, device=device)
    scheduler = UCBCurriculumScheduler(cfg["domains"])
    monitor   = SmokeTestMonitor(cfg)

    # ── Data loading ──────────────────────────────────────────────────────────
    print("[StrataRL M4] Loading datasets...")
    domain_data = {
        d: load_domain_samples(d, n=cfg["samples_per_domain"])
        for d in cfg["domains"]
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    wandb.init(project=cfg["wandb_project"], config=cfg, name="m4_smoke_50steps")
    
    current_temp = cfg["temperature"]

    for step in range(cfg["num_steps"]):
        phase = "bootstrap" if step < 30 else "strict"

        # Sample domain via UCB
        domain = scheduler.sample_domain()
        batch  = _sample_batch(domain_data[domain], cfg["batch_size"])

        prompts       = [item["prompt"]       for item in batch]
        ground_truths = [item["ground_truth"]  for item in batch]
        domains       = [domain] * len(batch)

        # Rollout
        rollouts = rollout_engine.generate(
            prompts,
            G              = cfg["G"],
            max_new_tokens = cfg["max_new_tokens"],
            min_new_tokens = cfg["min_new_tokens"],
            temperature    = current_temp,
        )
        
        # ── Temperature Hysteresis ───────────────────────────────────────────
        # Prevents oscillation while ensuring sufficient exploration
        unique_count = len(set(rollouts[0]["completions"]))
        diversity    = unique_count / cfg["G"]
        if diversity < 0.3:
            current_temp = min(1.0, current_temp + 0.05)
        elif diversity > 0.6:
            current_temp = max(cfg["temperature"], current_temp - 0.02)

        # Reward computation
        gdpo_rewards, raw_rewards = score_batch(rollouts, ground_truths, domains, phase=phase)

        # SAN advantages
        advantages = compute_san_advantages(gdpo_rewards, domains)

        # Token-level expansion
        comp_lengths = [[len(rollouts[i]["token_ids"][j]) for j in range(cfg["G"])]
                        for i in range(len(batch))]
        token_advs   = expand_advantages_to_tokens(advantages, comp_lengths)

        # Prepare tensors for policy update
        input_ids, attention_mask, completion_mask, old_logprobs = \
            _pack_rollouts(rollouts, tokenizer, device)

        # ── Token Accounting Assertion ────────────────────────────────────────
        # Verifies that advantages are correctly mapped to tokens after packing
        total_tokens_expected = sum(sum(l) for l in comp_lengths)
        total_tokens_actual   = completion_mask.sum().item()
        assert abs(total_tokens_expected - total_tokens_actual) < 5, \
            f"Token mismatch: expected {total_tokens_expected}, actual {total_tokens_actual}. Alignment drift!"

        token_adv_tensor = _expand_to_seq(token_advs, input_ids.shape, completion_mask)

        # GRPO loss
        model.train()
        optimizer.zero_grad()
        losses = grpo_loss(
            policy_model     = model,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            completion_mask  = completion_mask,
            advantages       = token_adv_tensor,
            old_logprobs     = old_logprobs,
            beta             = cfg["beta"],
            clip_eps         = cfg["clip_eps"],
        )
        losses["loss"].backward()
        
        # ── Gradient Diagnostics ──────────────────────────────────────────────
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                raise RuntimeError(f"NaN gradient detected in {name} at step {step}")
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        losses["grad_norm"] = grad_norm.item()
        
        optimizer.step()

        # Update curriculum correctly
        domain_advantages_map = defaultdict(list)
        for i, d in enumerate(domains):
            domain_advantages_map[d].extend(advantages[i].tolist())
        scheduler.update(domain_advantages_map)

        # ── Rollout Entropy (Sampled Behavior) ────────────────────────────────
        # Detects mode collapse before policy entropy shows it
        token_entropy_t = -torch.exp(old_logprobs) * old_logprobs
        losses["rollout_entropy"] = (token_entropy_t * completion_mask).sum().item() / (completion_mask.sum().item() + 1e-8)
        
        # Add diagnostic metrics for monitoring
        losses["advantage_std"] = advantages.std().item()
        losses["mean_abs_adv"]  = advantages.abs().mean().item()
        losses["gdpo_rewards"]  = gdpo_rewards
        losses["temperature"]   = current_temp

        # Monitoring
        alerts = monitor.log_step(step, losses, rollouts, raw_rewards, domain_advantages_map, phase=phase)
        if alerts:
            print(f"[Step {step}] ALERTS: {alerts}")

        if step % 5 == 0:
            print(f"[Step {step:3d}] loss={losses['loss']:.4f} "
                  f"raw_kl={losses['raw_kl_mean']:.4f} "
                  f"adv={losses['mean_abs_adv']:.4f} "
                  f"temp={current_temp:.2f} "
                  f"domain={domain}")

    print("\n✓ Smoke test completed (50 steps)")
    wandb.finish()


if __name__ == "__main__":
    run_smoke_test()
