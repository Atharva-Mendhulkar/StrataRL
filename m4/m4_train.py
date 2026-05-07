# m4/m4_train.py

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
from training.recompute         import should_recompute, teacher_forced_recompute, DRIFT_ABORT_THRESHOLD
from training.domain_guard      import assert_batch_domain_homogeneity
from curriculum.ucb_scheduler   import UCBCurriculumScheduler
from data.loaders               import load_domain_samples
from monitoring.monitor         import SmokeTestMonitor
from training.fallback          import FallbackController
from eval.benchmark_eval import BenchmarkEvaluator, BENCHMARKS


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
        start_idx  = r["completion_start_idx"]
        for i, comp_ids in enumerate(r["token_ids"]):
            full_ids = prompt_ids + comp_ids
            pad_len  = max_len - len(full_ids)
            
            all_input_ids.append(full_ids + [tokenizer.pad_token_id] * pad_len)
            all_att_masks.append([1] * len(full_ids) + [0] * pad_len)
            
            end_idx = r["completion_end_idxs"][i]
            comp_mask = [0] * max_len
            for idx in range(start_idx, end_idx):
                comp_mask[idx] = 1
            all_comp_masks.append(comp_mask)
            
            # Pack old_logprobs — I-8: must be zero in prompt region
            prompt_len = len(prompt_ids)
            # Use zeros for prompt, then rollout logprobs, then zeros for padding
            lps = [0.0] * prompt_len + r["rollout_logprobs"][i] + [0.0] * pad_len
            all_old_lps.append(lps)

    return (torch.tensor(all_input_ids, device=device),
            torch.tensor(all_att_masks, device=device),
            torch.tensor(all_comp_masks, device=device),
            torch.tensor(all_old_lps, device=device))


def _expand_to_seq(token_advs, shape, completion_mask):
    res = torch.zeros(shape, device=completion_mask.device)
    res[completion_mask == 1] = token_advs.to(completion_mask.device)
    return res


def run_smoke_test(config_path: str = "m4/m4_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"] if (torch.backends.mps.is_available() and cfg["device"] == "mps") else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16).to(device)
    model = get_peft_model(base_model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"]))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    rollout_engine = M4RolloutEngine(model, tokenizer, device=device)
    evaluator = BenchmarkEvaluator(generate_fn=rollout_engine.generate_for_eval)
    
    scheduler = UCBCurriculumScheduler(cfg["domains"])
    monitor   = SmokeTestMonitor(cfg)
    fallback_ctrl = FallbackController(cfg)

    domain_data = {d: load_domain_samples(d, n=cfg["samples_per_domain"]) for d in cfg["domains"]}
    wandb.init(project=cfg["wandb_project"], config=cfg, name="smoke_v3_empirical")
    
    current_temp = cfg["temperature"]

    for step in range(cfg["num_steps"]):
        phase = "bootstrap" if step < 30 else "strict"
        domain = scheduler.sample_domain()
        batch  = _sample_batch(domain_data[domain], cfg["batch_size"])
        prompts, gts = [i["prompt"] for i in batch], [i["ground_truth"] for i in batch]
        domains = [domain] * len(batch)
        
        # I-7: Batch-domain homogeneity assertion
        assert_batch_domain_homogeneity(domains)

        # Rollout
        rollouts = rollout_engine.generate(prompts, G=cfg["G"], temperature=current_temp)
        
        # I-4: Dynamic outcome weight from monitor
        w_outcome = monitor.delta_os_tracker.get_outcome_weight_override() or cfg.get("w_outcome", 0.7)
        w_struct  = 1.0 - w_outcome
        
        # I-1: score_batch with GDPO noise annealing
        combined_rewards, raw_rewards = score_batch(
            rollouts, gts, domains, 
            w_outcome=w_outcome, w_struct=w_struct,
            phase=phase, step=step
        )
        
        # I-5: SAN
        advantages = compute_san_advantages(combined_rewards, domains)
        
        comp_lengths = [[len(r["token_ids"][j]) for j in range(cfg["G"])] for r in rollouts]
        token_advs = expand_advantages_to_tokens(advantages, comp_lengths, use_length_norm=True)

        input_ids, attention_mask, completion_mask, old_logprobs = _pack_rollouts(rollouts, tokenizer, device)
        token_adv_tensor = _expand_to_seq(token_advs, input_ids.shape, completion_mask)

        model.train()
        losses = grpo_loss(
            model, input_ids, attention_mask, completion_mask, 
            token_adv_tensor, old_logprobs, 
            beta=cfg["beta"], clip_eps=cfg["clip_eps"]
        )
        
        optimizer.zero_grad()
        losses["loss"].backward()
        
        losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        # I-3: Teacher-forced recompute (periodic)
        if should_recompute(step):
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
                 fallback_ctrl.hard_stop(f"RECOMPUTE ABORT: drift={rc['recompute_drift_mean']:.4f}", step, optimizer, model, cfg)

        # Empirical Evaluation
        eval_interval = cfg.get("eval_interval", 100)
        if step % eval_interval == 0 and step > 0:
            eval_results = evaluator.run_all(step=step, greedy_only=True)
            for bench, r in eval_results.items():
                wandb.log({
                    f"eval/{bench}/greedy_acc":    r["greedy_acc"],
                    f"eval/{bench}/delta":         r["delta"],
                    f"eval/{bench}/target_met":    int(r["target_met"]),
                    f"eval/{bench}/avg_think_len": r["avg_think_len"],
                }, step=step)
            
            # MMLU negative control
            mmlu_acc = eval_results["mmlu"]["greedy_acc"]
            if mmlu_acc < BENCHMARKS["mmlu"]["baseline"] - 0.03:
                wandb.alert(
                    title=f"CATASTROPHIC FORGETTING: MMLU={mmlu_acc:.3f}",
                    text=f"MMLU dropped {(BENCHMARKS['mmlu']['baseline'] - mmlu_acc)*100:.1f}% below baseline.",
                    level=wandb.AlertLevel.ERROR,
                )

        # Update monitoring
        domain_adv_map = defaultdict(list)
        for i, d in enumerate(domains): domain_adv_map[d].extend(advantages[i].tolist())
        scheduler.update(domain_adv_map)
        
        alerts = monitor.log_step(step, losses, rollouts, raw_rewards, domain_adv_map, domains)
        
    wandb.finish()

if __name__ == "__main__":
    run_smoke_test()
