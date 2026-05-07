# monitoring/monitor.py

import wandb
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional

DELTA_OS_ALERT_THRESHOLD  = 0.05
DELTA_OS_WARN_THRESHOLD   = 0.15
DELTA_OS_WINDOW           = 100
DIVERSE_NONSENSE_TRIGGER  = 50


class DeltaOSTracker:
    def __init__(self):
        self.history           = deque(maxlen=DELTA_OS_WINDOW)
        self.steps_below_alert = 0
        self.attack_declared   = False
        self.intervention_count = 0

    def update(self, mean_outcome: float, mean_structural: float, step: int) -> dict:
        eps   = 1e-6
        delta = (mean_outcome + eps) / (mean_structural + eps)
        self.history.append(delta)
        rolling_mean = np.mean(list(self.history))

        if delta < DELTA_OS_ALERT_THRESHOLD:
            self.steps_below_alert += 1
        else:
            self.steps_below_alert = 0

        attack_active = self.steps_below_alert >= DIVERSE_NONSENSE_TRIGGER

        if attack_active and not self.attack_declared:
            self.attack_declared = True
            self.intervention_count += 1
            print(f"[Step {step}] DIVERSE NONSENSE ATTACK: Δ_O/S={delta:.4f}. "
                  f"Bumping w_outcome to 0.85.")
        elif not attack_active:
            self.attack_declared = False

        status = "ATTACK" if attack_active else \
                 "WARN"   if delta < DELTA_OS_WARN_THRESHOLD else "OK"

        return {
            "delta_os/value":         delta,
            "delta_os/rolling_mean":  rolling_mean,
            "delta_os/steps_below":   self.steps_below_alert,
            "delta_os/status":        status,
            "delta_os/interventions": self.intervention_count,
        }

    def get_outcome_weight_override(self) -> Optional[float]:
        return 0.85 if self.attack_declared else None


class SmokeTestMonitor:
    def __init__(self, config: dict):
        self.config = config
        self.delta_os_tracker = DeltaOSTracker()
        self.history = []

    def log_step(
        self,
        step:               int,
        losses:             Dict,
        rollouts:           List,
        raw_rewards:        torch.Tensor,   # [3, B, G]: [outcome, struct, token_rep]
        domain_advantages:  Dict,
        domains:            List[str],
    ) -> List[str]:
        outcome_r = raw_rewards[0]
        struct_r  = raw_rewards[1]
        
        mean_outcome = outcome_r.mean().item()
        mean_struct  = struct_r.mean().item()

        # I-4: Update DeltaOSTracker
        delta_os_metrics = self.delta_os_tracker.update(mean_outcome, mean_struct, step)
        
        metrics = {
            "train/loss":               losses.get("loss"),
            "train/raw_kl_mean":        losses.get("raw_kl_mean"),
            "train/entropy":            losses.get("entropy"),
            "train/clip_frac":          losses.get("clip_frac"),
            "learning/mean_outcome_reward":    mean_outcome,
            "learning/mean_structural_reward": mean_struct,
            "diversity/prefix_diversity":      self._compute_prefix_diversity(rollouts),
            "diversity/answer_entropy":        self._compute_answer_entropy(rollouts, domains),
        }
        metrics.update(delta_os_metrics)
        
        wandb.log(metrics, step=step)

        # Alerts
        alerts = []
        if delta_os_metrics["delta_os/status"] == "ATTACK":
            alerts.append("DIVERSE_NONSENSE_ATTACK")
            
        if losses.get("raw_kl_mean", 0) > 0.10:
            alerts.append("KL_DIVERGENCE_CRITICAL: policy drifting too fast")

        if losses.get("entropy", 1.0) < 0.05:
            alerts.append("ENTROPY_COLLAPSE")

        if metrics["diversity/prefix_diversity"] < 0.20:
            alerts.append("BLE_WARNING: prefix_diversity < 0.20")

        for alert in alerts:
            print(f"[Step {step:4d}] ALERT: {alert}")

        self.history.append(metrics)
        return alerts

    def _compute_answer_entropy(self, rollouts: List, domains: List[str]) -> float:
        """H_answer: entropy over extracted normalized answers in group G."""
        from rewards.outcome_verifiers import extract_answer
        from collections import Counter
        import math

        all_entropies = []
        for rollout in rollouts:
            answers = []
            for completion in rollout["completions"]:
                ans = extract_answer(completion)
                answers.append(str(ans).strip().lower() if ans else "__NONE__")

            counts = Counter(answers)
            total  = sum(counts.values())
            if total < 2:
                continue
            entropy = -sum((c/total) * math.log(c/total + 1e-9)
                          for c in counts.values())
            all_entropies.append(entropy)

        return float(np.mean(all_entropies)) if all_entropies else 0.0

    def _compute_prefix_diversity(self, rollouts: List, n_tokens: int = 50) -> float:
        unique_fracs = []
        for rollout in rollouts:
            prefixes = set()
            for token_ids in rollout.get("token_ids", []):
                prefix = tuple(token_ids[:n_tokens]) if len(token_ids) >= n_tokens else tuple(token_ids)
                prefixes.add(prefix)
            G = len(rollout.get("token_ids", [1]))
            unique_fracs.append(len(prefixes) / G)
        return float(np.mean(unique_fracs)) if unique_fracs else 0.0
