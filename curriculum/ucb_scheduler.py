# curriculum/ucb_scheduler.py

import numpy as np
from typing import List, Dict

class UCBCurriculumScheduler:
    """
    Multi-Armed Bandit (UCB) scheduler for domain selection.
    
    Prioritizes domains where:
      1. Advantages are high (learning is happening)
      2. Uncertainty/exploration is high (not sampled enough)
    
    Now supports collapse resistance (I-4 integration).
    """
    def __init__(self, domains: List[str], exploration_weight: float = 1.0):
        self.domains = domains
        self.c       = exploration_weight
        self.counts  = {d: 1 for d in domains}
        self.scores  = {d: 0.5 for d in domains}
        self.total_steps = len(domains)
        self.collapse_types = {d: "HEALTHY" for d in domains}
        self.domain_sample_counts = {d: 0 for d in domains}

    def sample_domain(self) -> str:
        """Sample a domain using UCB values as probabilities (or greedy)."""
        weights = self.get_weights()
        domains = list(weights.keys())
        probs   = list(weights.values())
        
        # Stochastic sampling based on UCB weights
        chosen = np.random.choice(domains, p=probs)
        return chosen

    MIN_DOMAIN_WEIGHT = {
        "strategyqa": 0.15,
        "gsm8k":      0.10,
        "mmlu":       0.10,
    }

    def get_weights(self) -> Dict[str, float]:
        """Compute sampling weights based on UCB and collapse status."""
        ucb_values = {}
        for d in self.domains:
            # UCB formula
            exploitation = self.scores[d]
            exploration  = self.c * np.sqrt(np.log(self.total_steps) / self.counts[d])
            combined = exploitation + exploration
            
            # Adaptive Exploration Floor
            floor = 0.05 * (1 + 0.1 * np.log(self.total_steps + 1))
            val   = max(combined, floor)
            
            # Collapse suppression
            if self.collapse_types[d] == "ALL_CORRECT":
                val *= 0.01  # Suppress finished domains
            elif self.collapse_types[d] == "ALL_WRONG":
                val *= 0.5   # Reduce weight for broken domains to allow recovery check
                
            ucb_values[d] = val

        for domain, floor in self.MIN_DOMAIN_WEIGHT.items():
            if domain in ucb_values:
                ucb_values[domain] = max(ucb_values[domain], floor)

        # Softmax or simple normalization to get probabilities
        total = sum(ucb_values.values())
        return {d: v / total for d, v in ucb_values.items()}

    def set_collapse_types(self, collapse_types: Dict[str, str]):
        """Update collapse status from monitor."""
        self.collapse_types.update(collapse_types)

    def update(self, domain_advantages: Dict[str, List[float]]):
        """Update domain statistics with new advantages."""
        for d, advs in domain_advantages.items():
            if not advs:
                continue
            
            # Use mean absolute advantage as a proxy for 'learning signal'
            signal = np.mean(np.abs(advs))
            
            # Incremental update
            self.counts[d] += 1
            self.total_steps += 1
            alpha = 1.0 / self.counts[d]
            self.scores[d] = (1 - alpha) * self.scores[d] + alpha * signal
            
            # Update sample count (for tests)
            self.domain_sample_counts[d] += len(advs)
