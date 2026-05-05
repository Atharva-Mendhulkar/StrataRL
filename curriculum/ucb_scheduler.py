import numpy as np
from typing import List, Dict

class UCBCurriculumScheduler:
    """
    Multi-Armed Bandit (UCB) scheduler for domain selection.
    
    Prioritizes domains where:
      1. Advantages are high (learning is happening)
      2. Uncertainty/exploration is high (not sampled enough)
    """
    def __init__(self, domains: List[str], exploration_weight: float = 1.0):
        self.domains = domains
        self.c       = exploration_weight
        self.counts  = {d: 1 for d in domains}
        self.scores  = {d: 0.5 for d in domains}
        self.total_steps = len(domains)

    def sample_domain(self) -> str:
        """Sample a domain using UCB."""
        ucb_values = {}
        for d in self.domains:
            exploitation = self.scores[d]
            exploration  = self.c * np.sqrt(np.log(self.total_steps) / self.counts[d])
            
            combined = exploitation + exploration
            
            # ── Adaptive Exploration Floor ────────────────────────────────────
            # Keeps exploration alive longer as training progresses to avoid early 
            # domain starvation or random oscillations.
            floor = 0.05 * (1 + 0.1 * np.log(self.total_steps + 1))
            ucb_values[d] = max(combined, floor)

        # Greedy choice over UCB values
        chosen = max(ucb_values, key=ucb_values.get)
        return chosen

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
