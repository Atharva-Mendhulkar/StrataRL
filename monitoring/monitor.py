import wandb
import numpy as np
import torch
from typing import Dict, List

class SmokeTestMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.history = []
        self.low_adv_count = 0

    def log_step(self, step: int, losses: Dict, rollouts: List, raw_rewards: torch.Tensor, domain_advantages: Dict, phase: str = "strict"):
        # raw_rewards: [3, B, G] -> [outcome, struct, token_rep]
        outcome_r = raw_rewards[0]
        struct_r  = raw_rewards[1]
        
        # Calculate diagnostics
        outcome_std = outcome_r.std().item()
        struct_std  = struct_r.std().item()
        
        # Fraction of rewards that are absolute zero (failed both structure and outcome)
        zero_frac = (raw_rewards[:2].sum(dim=0) == 0).float().mean().item()
        
        gdpo_rewards = losses.get("gdpo_rewards", torch.zeros(1))
        pos_ratio = (gdpo_rewards > 0).float().mean().item()
        
        # ── Rollout Diversity & Behavior ──────────────────────────────────────
        unique_completions = len(set(rollouts[0]["completions"]))
        G = len(rollouts[0]["completions"])
        prefix_diversity = unique_completions / G

        metrics = {
            "step": step,
            "train/loss":               losses["loss"].item(),
            "train/policy_loss":         losses["policy_loss"].item(),
            "train/kl":                  losses["kl"].item(),
            "train/raw_kl_mean":         losses.get("raw_kl_mean", 0.0),
            "train/entropy":             losses["entropy"].item(),
            "train/rollout_entropy":     losses.get("rollout_entropy", 0.0),
            "train/ratio_clipped_frac":  losses.get("ratio_clipped_frac", 0.0),
            "train/grad_norm":           losses.get("grad_norm", 0.0),
            
            "learning/outcome_reward_mean": outcome_r.mean().item(),
            "learning/outcome_reward_std":  outcome_std,
            "learning/struct_reward_mean":  struct_r.mean().item(),
            "learning/struct_reward_std":   struct_std,
            "learning/fraction_zero_rewards": zero_frac,
            "learning/reward_positive_ratio": pos_ratio,
            "learning/prefix_diversity":     prefix_diversity,
            "learning/mean_abs_advantage":   losses.get("mean_abs_adv", 0.0),
            "learning/advantage_std":        losses.get("advantage_std", 0.0),
            "meta/phase_bootstrap":          1 if phase == "bootstrap" else 0,
        }
        
        # Add domain specific metrics
        for domain, advs in domain_advantages.items():
            if advs:
                metrics[f"curriculum/{domain}/mean_adv"] = np.mean(advs)

        wandb.log(metrics, step=step)
        self.history.append(metrics)
        
        alerts = []
        if losses["kl"].item() < 0:
            alerts.append("KL_NEGATIVE")
        if losses["entropy"].item() < 0.5 and step > 20:
            alerts.append("ENTROPY_COLLAPSE")
        
        # NO_LEARNING_SIGNAL alert
        adv_std = losses.get("advantage_std", 0.0)
        if adv_std < 0.05:
            self.low_adv_count += 1
        else:
            self.low_adv_count = 0
            
        if self.low_adv_count >= 5:
            alerts.append("NO_LEARNING_SIGNAL")
            
        return alerts
