import pytest
import torch
from training.advantage import compute_san_advantages, expand_advantages_to_tokens

def _global_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)

class TestSAN:

    def test_single_domain_zero_mean(self):
        """Within a single domain, normalized advantages must have zero mean."""
        rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0],
                                 [0.5, 0.5, 0.5, 0.5]])
        domains = ["gsm8k", "gsm8k"]
        adv     = compute_san_advantages(rewards, domains)
        # All values from same domain: mean should be ~0
        assert abs(adv.mean().item()) < 1e-4

    def test_multi_domain_independent_normalization(self):
        """
        Critical test: high-reward domain should NOT penalize low-reward domain.

        Scenario: GSM8K completions all score 0.9 (easy domain, high rewards)
                  StrategyQA completions all score 0.4 (hard domain, lower rewards)

        Under GLOBAL normalization: StrategyQA gets negative advantage (WRONG)
        Under SAN: each domain normalized independently (CORRECT)
        """
        rewards = torch.tensor([
            [0.9, 0.9, 0.8, 0.9],   # gsm8k — high rewards
            [0.9, 0.9, 0.8, 0.9],   # gsm8k
            [0.4, 0.3, 0.4, 0.5],   # strategyqa — lower rewards
            [0.4, 0.3, 0.4, 0.5],   # strategyqa
        ])
        domains = ["gsm8k", "gsm8k", "strategyqa", "strategyqa"]

        adv_san    = compute_san_advantages(rewards, domains)
        adv_global = _global_normalize(rewards)   # what standard GRPO does

        # Under SAN: StrategyQA completions should have mixed sign advantages
        # (some positive, some negative — relative to their own domain)
        strat_adv_san = adv_san[2:4]
        assert strat_adv_san.std().item() > 0.1, \
            "StrategyQA should have variance in SAN advantages"

        # Under global normalization: StrategyQA completions get ALL negative advantages
        # (they score below the global mean dominated by GSM8K)
        strat_adv_global = adv_global[2:4]
        assert strat_adv_global.mean().item() < -0.5, \
            "Global norm unfairly penalizes StrategyQA — this is the bug SAN fixes"

        # SAN StrategyQA mean should be near zero (fair treatment)
        assert abs(strat_adv_san.mean().item()) < 0.1, \
            "SAN should give StrategyQA ~zero mean advantage"

    def test_advantage_clipping(self):
        """No advantage should exceed ±5.0."""
        rewards = torch.zeros(4, 8)
        rewards[0, 0] = 100.0    # extreme outlier
        domains = ["gsm8k"] * 4
        adv = compute_san_advantages(rewards, domains)
        assert adv.max().item() <= 5.0
        assert adv.min().item() >= -5.0

    def test_single_stratum_zero_advantage(self):
        """Single sample in a stratum cannot be compared — must get zero advantage."""
        rewards = torch.tensor([[0.8, 0.9, 0.7, 0.85]])
        domains = ["rare_domain"]
        adv = compute_san_advantages(rewards, domains)
        assert (adv == 0.0).all()

    def test_zero_variance_stratum_no_nan(self):
        """All G completions have same reward → σ=0 → must not produce NaN."""
        rewards = torch.ones(2, 4) * 0.7
        domains = ["gsm8k", "gsm8k"]
        adv = compute_san_advantages(rewards, domains)
        assert not torch.isnan(adv).any()


class TestLengthNorm:

    def test_long_completion_clamped(self):
        """Completions > 512 tokens get same normalization factor."""
        advantages = torch.tensor([[1.0, 1.0]])
        lengths    = [[600, 800]]   # both > 512
        expanded   = expand_advantages_to_tokens(advantages, lengths)
        # Both get factor 1/sqrt(512) — they should have same per-token advantage
        adv_600 = expanded[:600].mean().item()
        adv_800 = expanded[600:].mean().item()
        assert abs(adv_600 - adv_800) < 1e-5, \
            "Completions > 512 tokens must get identical length normalization"

    def test_short_completion_penalized_less(self):
        """
        Longer completions get smaller per-token advantage (penalized for length).
        But clamped so very long completions don't get too small.
        """
        advantages = torch.tensor([[1.0, 1.0]])
        lengths    = [[100, 400]]
        expanded   = expand_advantages_to_tokens(advantages, lengths)
        adv_100 = expanded[:100].mean().item()
        adv_400 = expanded[100:].mean().item()
        # factor 1/sqrt(100) = 0.1, 1/sqrt(400) = 0.05
        assert adv_100 > adv_400
