import pytest
import numpy as np
from curriculum.ucb_scheduler import UCBCurriculumScheduler


class TestUCBScheduler:

    def test_initial_weights_uniform(self):
        """Before any updates, all domains should have equal sampling probability."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu", "strategyqa"])
        weights   = scheduler.get_weights()
        vals      = list(weights.values())
        assert max(vals) - min(vals) < 0.1, "Initial weights should be approximately uniform"

    def test_weights_sum_to_one(self):
        """Sampling probabilities must sum to 1.0."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu", "strategyqa"])
        weights   = scheduler.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_correct_domain_suppressed(self):
        """ALL_CORRECT domain should get zero probability."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu"])
        scheduler.set_collapse_types({"gsm8k": "ALL_CORRECT", "mmlu": "HEALTHY"})
        weights   = scheduler.get_weights()
        assert weights["gsm8k"] < 0.01, \
            "ALL_CORRECT domain must be suppressed to near-zero"
        assert weights["mmlu"] > 0.99, \
            "Non-collapsed domain must absorb ALL_CORRECT's probability"

    def test_all_wrong_domain_reduced_not_zero(self):
        """ALL_WRONG domain should be suppressed but not eliminated."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu", "strategyqa"])
        scheduler.set_collapse_types({
            "gsm8k": "ALL_WRONG",
            "mmlu": "HEALTHY",
            "strategyqa": "HEALTHY"
        })
        weights = scheduler.get_weights()
        assert weights["gsm8k"] > 0.01, \
            "ALL_WRONG domain must retain some probability (not eliminated)"
        assert weights["gsm8k"] < weights["mmlu"], \
            "ALL_WRONG domain must be less probable than healthy domain"

    def test_exploration_bonus_for_undersampled(self):
        """Domain sampled fewer times should get exploration bonus."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu"])
        # Update gsm8k many times
        for _ in range(50):
            scheduler.update({"gsm8k": [0.5, 0.3, 0.6, 0.4]})
            scheduler.domain_sample_counts["gsm8k"] += 4

        weights = scheduler.get_weights()
        # mmlu has been sampled only once (init) — should get exploration bonus
        assert weights["mmlu"] > 0.2, \
            "Undersampled domain should receive exploration bonus from UCB"

    def test_no_crash_all_domains_collapsed(self):
        """Emergency reset: if all domains collapse, must not crash."""
        scheduler = UCBCurriculumScheduler(["gsm8k", "mmlu"])
        scheduler.set_collapse_types({"gsm8k": "ALL_CORRECT", "mmlu": "ALL_CORRECT"})
        weights = scheduler.get_weights()   # must not raise
        assert abs(sum(weights.values()) - 1.0) < 1e-6
