import torch
import torch.nn.functional as F
import wandb
import yaml
import random
import numpy as np
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
    all_input_ids, all_att_masks, all_comp_masks, all_old_lps = [], [], [], []
    max_len = 0
    for r in rollouts:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=True).input_ids
        for comp_ids in r["token_ids"]:
            max_len = max(max_len, len(prompt_ids) + len(comp_ids))

    for r in rollouts:
        prompt_ids = tokenizer(r["prompt"], add_special_tokens=True).input_ids
        for i, comp_ids in enumerate(r["token_ids"]):
            full_ids = prompt_ids + comp_ids
            pad_len  = max_len - len(full_ids)
            all_input_ids.append(full_ids + [tokenizer.pad_token_id] * pad_len)
            all_att_masks.append([1] * len(full_ids) + [0] * pad_len)
            all_comp_masks.append([0] * len(prompt_ids) + [1] * len(comp_ids) + [0] * pad_len)
            all_old_lps.append([0.0] * len(prompt_ids) + r["rollout_logprobs"][i] + [0.0] * pad_len)

    input_ids_t    = torch.tensor(all_input_ids, device=device)
    attention_mask_t = torch.tensor(all_att_masks, device=device)
    completion_mask_t = torch.tensor(all_comp_masks, device=device)
    old_logprobs_t = torch.tensor(all_old_lps, device=device)

    assert old_logprobs_t.shape == input_ids_t.shape, "Packing alignment failure"
    return input_ids_t, attention_mask_t, completion_mask_t, old_logprobs_t


def _expand_to_seq(token_advs, shape, completion_mask):
    res = torch.zeros(shape, device=completion_mask.device)
    res[completion_mask == 1] = token_advs.to(completion_mask.device)
    return res


def run_smoke_test(config_path: str = "m4/m4_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"] if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16).to(device)
    model = get_peft_model(base_model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"]))
    
    print({"device": device, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "G": cfg["G"], "beta": cfg["beta"]})

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    rollout_engine = M4RolloutEngine(model, tokenizer, device=device)
    scheduler = UCBCurriculumScheduler(cfg["domains"])
    monitor   = SmokeTestMonitor(cfg)

    domain_data = {d: load_domain_samples(d, n=cfg["samples_per_domain"]) for d in cfg["domains"]}

    # ── Step 0: Curriculum Metadata Sanity Check ──────────────────────────────
    print("[Audit] Performing Step 0 curriculum sanity check...")
    test_batch = _sample_batch(domain_data["gsm8k"], 5)
    test_rollouts = rollout_engine.generate([item["prompt"] for item in test_batch], G=1)
    # verify base model can at least generate tokens (non-empty)
    if any(len(c) == 0 for r in test_rollouts for c in r["completions"]):
        raise RuntimeError("Curriculum Audit Failed: Base model producing empty completions.")

    wandb.init(project=cfg["wandb_project"], config=cfg, name="m4_hardened_run")
    
    current_temp = cfg["temperature"]
    length_coeff = 0.01
    
    # State for GDPO cooldown
    last_noise_step = -100
    avg_length_history = []
    avg_outcome_history = []

    for step in range(cfg["num_steps"]):
        phase = "bootstrap" if step < 30 else "strict"
        domain = scheduler.sample_domain()
        batch  = _sample_batch(domain_data[domain], cfg["batch_size"])
        prompts, gts = [i["prompt"] for i in batch], [i["ground_truth"] for i in batch]

        # Rollout
        rollouts = rollout_engine.generate(prompts, G=cfg["G"], temperature=current_temp)
        
        # Adaptive Temperature Hysteresis
        unique_count = len(set(rollouts[0]["completions"]))
        diversity = unique_count / cfg["G"]
        if diversity < 0.3: current_temp = min(1.0, current_temp + 0.05)
        elif diversity > 0.6: current_temp = max(cfg["temperature"], current_temp - 0.02)

        # Reward computation with GDPO Cooldown (lock for 10 steps after noise injection)
        cooldown_active = (step - last_noise_step) < 10
        gdpo_rewards, raw_rewards = score_batch(rollouts, gts, [domain]*len(batch), phase=phase, gdpo_cooldown=cooldown_active)
        
        # Track if noise was actually injected (detect via advantage std)
        if gdpo_rewards.std() < 1e-3 and not cooldown_active:
             last_noise_step = step

        # SAN & Advantage Expansion
        advantages = compute_san_advantages(gdpo_rewards, [domain]*len(batch))
        
        # ── Verbosity Monitor & Corrective Action ─────────────────────────────
        current_avg_len = np.mean([[len(r["token_ids"][j]) for j in range(cfg["G"])] for r in rollouts])
        current_avg_out = raw_rewards[0].mean().item()
        avg_length_history.append(current_avg_len)
        avg_outcome_history.append(current_avg_out)
        
        if step > 20:
            len_growth = current_avg_len / (np.mean(avg_length_history[:10]) + 1e-8)
            out_growth = (current_avg_out + 1e-4) / (np.mean(avg_outcome_history[:10]) + 1e-4)
            if len_growth > 1.2 and out_growth < 1.05:
                length_coeff = min(0.05, length_coeff + 0.005) # Aggressive dampening

        comp_lengths = [[len(r["token_ids"][j]) for j in range(cfg["G"])] for r in rollouts]
        token_advs = expand_advantages_to_tokens(advantages, comp_lengths, length_norm_coeff=length_coeff)

        # Policy Update
        input_ids, attention_mask, completion_mask, old_logprobs = _pack_rollouts(rollouts, tokenizer, device)
        token_adv_tensor = _expand_to_seq(token_advs, input_ids.shape, completion_mask)

        model.train()
        optimizer.zero_grad()
        losses = grpo_loss(model, input_ids, attention_mask, completion_mask, token_adv_tensor, old_logprobs, beta=cfg["beta"], clip_eps=cfg["clip_eps"])
        
        # ── Numerical Invariant Audit ─────────────────────────────────────────
        if step % 25 == 0:
             with torch.no_grad():
                 logits = model(input_ids).logits
                 recomputed_logp = F.log_softmax(logits, dim=-1)
                 # Sample one answer token to verify numerical consistency
                 row, col = (completion_mask == 1).nonzero()[0]
                 recomputed = recomputed_logp[row, col-1, input_ids[row, col]].item()
                 old_lp = old_logprobs[row, col].item()
                 if abs(recomputed - old_lp) > 1e-3:
                     print(f"[Warning] Numerical drift: {recomputed:.5f} vs {old_lp:.5f}")

        losses["loss"].backward()
        
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                raise RuntimeError(f"NaN grad in {name}")
        
        losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        # Curriculum & Monitoring
        domain_adv_map = defaultdict(list)
        for i, d in enumerate([domain]*len(batch)): domain_adv_map[d].extend(advantages[i].tolist())
        scheduler.update(domain_adv_map)
        
        losses.update({"advantage_std": advantages.std().item(), "mean_abs_adv": advantages.abs().mean().item(), "gdpo_rewards": gdpo_rewards, "length_coeff": length_coeff})
        monitor.log_step(step, losses, rollouts, raw_rewards, domain_adv_map, phase=phase)

    print("\n✓ Hardened training completed")
    wandb.finish()

if __name__ == "__main__":
    run_smoke_test()
