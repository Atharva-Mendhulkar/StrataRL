"""
Kaggle production training loop for EXP_01.
Equivalent to m4/m4_train.py but:
  - Uses KaggleRolloutEngine (CUDA + BitsAndBytes 4-bit) instead of M4RolloutEngine
  - Uses Unsloth LoRA for 2x faster training
  - Configures grad_accum for P100 memory constraints
  - Adds vllm_sync_interval periodic syncing placeholder
"""

import torch
import yaml
import random
import wandb
import argparse
import numpy as np
import os
from collections import defaultdict
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer

from engines.kaggle_rollout_engine    import build_kaggle_engine
from rewards.reward_engine           import score_batch
from training.advantage              import compute_san_advantages, expand_advantages_to_tokens
from training.policy_update          import grpo_loss
from training.recompute              import should_recompute, teacher_forced_recompute, DRIFT_ABORT_THRESHOLD
from training.domain_guard           import assert_batch_domain_homogeneity
from curriculum.ucb_scheduler        import UCBCurriculumScheduler
from data.loaders                    import load_domain_samples
from monitoring.monitor              import SmokeTestMonitor
from training.fallback               import FallbackController
from eval.benchmark_eval             import BenchmarkEvaluator, BENCHMARKS


def _sample_batch(samples, batch_size):
    return random.sample(samples, min(batch_size, len(samples)))


def _pack_rollouts(rollouts, tokenizer, device):
    all_input_ids, all_att_masks, all_comp_masks, all_old_lps = [], [], [], []
    max_len = 0
    
    def _tokenize_prompt(prompt):
        if isinstance(prompt, list):
            prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            return tokenizer(prompt_str, add_special_tokens=False).input_ids
        return tokenizer(prompt, add_special_tokens=True).input_ids

    for r in rollouts:
        prompt_ids = _tokenize_prompt(r["prompt"])
        for comp_ids in r["token_ids"]:
            max_len = max(max_len, len(prompt_ids) + len(comp_ids))

    for r in rollouts:
        prompt_ids = _tokenize_prompt(r["prompt"])
        start_idx  = r["completion_start_idx"]
        for i, comp_ids in enumerate(r["token_ids"]):
            full_ids = prompt_ids + comp_ids
            pad_len  = max_len - len(full_ids)

            all_input_ids.append(full_ids + [tokenizer.pad_token_id] * pad_len)
            all_att_masks.append([1] * len(full_ids) + [0] * pad_len)

            end_idx   = r["completion_end_idxs"][i]
            comp_mask = [0] * max_len
            for idx in range(start_idx, end_idx):
                comp_mask[idx] = 1
            all_comp_masks.append(comp_mask)

            # I-8: prompt region must be zero in old_logprobs
            prompt_len = len(prompt_ids)
            lps = [0.0] * prompt_len + r["rollout_logprobs"][i] + [0.0] * pad_len
            all_old_lps.append(lps)

    return (
        torch.tensor(all_input_ids, device=device),
        torch.tensor(all_att_masks, device=device),
        torch.tensor(all_comp_masks, device=device),
        torch.tensor(all_old_lps, device=device),
    )


def _expand_to_seq(token_advs, shape, completion_mask):
    res = torch.zeros(shape, device=completion_mask.device)
    res[completion_mask == 1] = token_advs.to(completion_mask.device)
    return res


def run_kaggle_training(config_path: str, run_name: str = None, wandb_project: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if run_name:
        cfg["run_name"] = run_name
    if wandb_project:
        cfg["wandb_project"] = wandb_project

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available — are you on a GPU instance?"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build engine with 4-bit quantization for P100 (16GB VRAM)
    engine = build_kaggle_engine(
        model_id     = cfg["model_id"],
        load_in_4bit = cfg.get("load_in_4bit", True),
    )
    tokenizer = engine.tokenizer

    # Prepare base model for quantized training:
    #   - Casts layernorm to float32 (prevents NaN with 4-bit)
    #   - Enables gradient checkpointing with use_reentrant=False
    #     (use_reentrant=True crashes with frozen/quantized base layers)
    #   - Enables input embedding gradients for gradient flow
    base_model = prepare_model_for_kbit_training(
        engine.model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = cfg["lora_r"],
        lora_alpha     = cfg["lora_alpha"],
        target_modules = cfg["target_modules"],
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Update engine so rollouts use the LoRA-adapted model
    engine.model = model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(cfg.get("lr", 5e-6)),
        weight_decay = 0.01,
    )

    grad_accum = cfg.get("grad_accum", 4)

    evaluator     = BenchmarkEvaluator(generate_fn=engine.generate_for_eval)
    scheduler     = UCBCurriculumScheduler(cfg["domains"], exploration_weight=cfg.get("ucb_c", 0.5))
    monitor       = SmokeTestMonitor(cfg)
    fallback_ctrl = FallbackController(cfg)

    domain_data = {d: load_domain_samples(d, n=cfg.get("samples_per_domain", 500))
                   for d in cfg["domains"]}

    wandb.init(
        project = cfg.get("wandb_project", "stratarl_kaggle_3b"),
        config  = cfg,
        name    = cfg.get("run_name", "EXP_01_qwen3b"),
    )

    current_temp = cfg.get("temperature", 0.85)
    optimizer.zero_grad()

    for step in range(cfg["num_steps"]):
        phase  = "bootstrap" if step < 30 else "strict"
        domain = scheduler.sample_domain()
        batch  = _sample_batch(domain_data[domain], cfg["batch_size"])
        prompts, gts = [i["prompt"] for i in batch], [i["ground_truth"] for i in batch]
        domains = [domain] * len(batch)

        # I-7: batch homogeneity
        assert_batch_domain_homogeneity(domains)

        rollouts = engine.generate(
            prompts,
            G              = cfg["G"],
            temperature    = current_temp,
            max_new_tokens = cfg.get("max_new_tokens", 2048),
            min_new_tokens = cfg.get("min_new_tokens", 100),
        )

        # Free rollout VRAM before training forward pass
        torch.cuda.empty_cache()

        # I-4: dynamic outcome weight
        w_outcome = monitor.delta_os_tracker.get_outcome_weight_override() or cfg.get("w_outcome", 0.7)
        w_struct  = 1.0 - w_outcome

        # I-1: score with GDPO noise annealing
        combined_rewards, raw_rewards = score_batch(
            rollouts, gts, domains,
            w_outcome = w_outcome,
            w_struct  = w_struct,
            phase     = phase,
            step      = step,
        )

        # I-5: SAN vs Global Advantages
        use_san = cfg.get("use_san", True)
        if use_san:
            advantages = compute_san_advantages(combined_rewards, domains)
        else:
            from training.advantage import compute_global_advantages
            advantages = compute_global_advantages(combined_rewards)
            
        comp_lengths = [[len(r["token_ids"][j]) for j in range(cfg["G"])] for r in rollouts]
        token_advs   = expand_advantages_to_tokens(advantages, comp_lengths, use_length_norm=False)

        input_ids, attention_mask, completion_mask, old_logprobs = _pack_rollouts(
            rollouts, tokenizer, device
        )
        token_adv_tensor = _expand_to_seq(token_advs, input_ids.shape, completion_mask)

        model.train()
        losses = grpo_loss(
            model, input_ids, attention_mask, completion_mask,
            token_adv_tensor, old_logprobs,
            beta     = cfg.get("beta", 0.01),
            clip_eps = cfg.get("clip_eps", 0.2),
        )

        # Gradient accumulation
        (losses["loss"] / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            optimizer.zero_grad()

        # I-3: teacher-forced recompute
        recompute_interval = cfg.get("recompute_interval", 25)
        if step % recompute_interval == 0:
            rc = teacher_forced_recompute(
                policy_model    = model,
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                completion_mask = completion_mask,
                old_logprobs    = old_logprobs,
                step            = step,
            )
            losses.update(rc)
            wandb.log(rc, step=step)

            if rc["recompute_status"] == "ABORT":
                fallback_ctrl.hard_stop(
                    f"RECOMPUTE ABORT: drift={rc['recompute_drift_mean']:.4f}",
                    step, optimizer, model, cfg,
                )

        # Empirical eval
        eval_interval = cfg.get("eval_interval", 100)
        if step % eval_interval == 0 and step > 0:
            eval_n = cfg.get("eval_n_samples", 20)
            eval_results = evaluator.run_all(step=step, greedy_only=True, n_samples=eval_n)
            for bench, r in eval_results.items():
                delta = r["greedy_acc"] - r["baseline_lit"]
                wandb.log({
                    f"eval/{bench}/greedy_acc":    r["greedy_acc"],
                    f"eval/{bench}/delta":         delta,
                    f"eval/{bench}/target_met":    int(delta >= 0),
                    f"eval/{bench}/avg_think_len": r["avg_think_len"],
                }, step=step)

            # MMLU negative control — catastrophic forgetting gate
            mmlu_acc = eval_results.get("mmlu", {}).get("greedy_acc", 1.0)
            mmlu_baseline = BENCHMARKS["mmlu"]["baseline_lit"]
            if mmlu_acc < mmlu_baseline - 0.03:
                wandb.alert(
                    title=f"CATASTROPHIC FORGETTING: MMLU={mmlu_acc:.3f}",
                    text=f"MMLU dropped below baseline by {(mmlu_baseline - mmlu_acc)*100:.1f}%.",
                    level=wandb.AlertLevel.ERROR,
                )

        # Save checkpoint every 100 steps
        if step % 100 == 0 and step > 0:
            ckpt_dir = f"outputs/step_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[Step {step}] Checkpoint saved to {ckpt_dir}")

        # Monitor
        domain_adv_map = defaultdict(list)
        for i, d in enumerate(domains):
            domain_adv_map[d].extend(advantages[i].tolist())
        scheduler.update(domain_adv_map)

        alerts = monitor.log_step(step, losses, rollouts, raw_rewards, domain_adv_map, domains)

    # Final save
    os.makedirs("outputs/final", exist_ok=True)
    model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")
    print("Training complete. Final adapter saved to outputs/final/")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/exp_01_kaggle.yaml")
    parser.add_argument("--run_name",       default=None)
    parser.add_argument("--wandb_project",  default=None)
    args = parser.parse_args()
    run_kaggle_training(args.config, args.run_name, args.wandb_project)
