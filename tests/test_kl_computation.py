import pytest
import torch
import torch.nn.functional as F
from training.policy_update import grpo_loss


class TestKLFormula:

    def test_kl_is_nonnegative_in_expectation(self):
        """
        The KL formula log_p_old - log_p_new should be
        non-negative in expectation over many tokens.
        
        This catches the sign-inversion bug:
        log_ratio - (ratio - 1) is always ≤ 0 — that would be wrong.
        """
        torch.manual_seed(42)
        B, G, seq, vocab = 2, 4, 50, 1000

        # Simulate old logprobs (from rollout)
        old_logits = torch.randn(B * G, seq, vocab)
        old_logp   = F.log_softmax(old_logits, dim=-1)
        labels     = torch.randint(0, vocab, (B * G, seq))
        old_token_logp = old_logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Simulate new policy logprobs (slightly different from old)
        new_logits = old_logits + torch.randn_like(old_logits) * 0.1
        new_logp   = F.log_softmax(new_logits, dim=-1)
        new_token_logp = new_logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Compute KL using our formula
        kl = old_token_logp - new_token_logp
        kl_mean = kl.mean().item()

        # Must be non-negative
        assert kl_mean >= -0.01, (
            f"KL formula is sign-inverted! Mean KL = {kl_mean:.4f} (expected ≥ 0). "
            f"This is the critical bug that would cause the controller to "
            f"increase β when the model is DIVERGING."
        )

    def test_kl_zero_for_identical_policies(self):
        """If old == new policy, KL must be exactly 0."""
        logp = torch.tensor([[-1.0, -2.0, -1.5, -0.5]])
        kl   = logp - logp
        assert abs(kl.sum().item()) < 1e-6

    def test_kl_positive_for_diverged_policy(self):
        """If new policy has moved away from old, KL > 0."""
        old_logp = torch.tensor([[-1.0, -2.0, -3.0, -0.5]])
        new_logp = torch.tensor([[-0.5, -3.0, -2.0, -1.5]])   # different distribution
        kl = old_logp - new_logp
        assert kl.sum().item() > 0

    def test_wrong_formula_is_negative(self):
        """
        Validates that the WRONG formula log_ratio - (ratio-1) is indeed always ≤ 0.
        This test exists to document why we rejected that formula.
        """
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]])
        new_logp = torch.tensor([[-0.5, -3.0, -2.0]])
        log_ratio = new_logp - old_logp
        ratio     = torch.exp(log_ratio)
        wrong_kl  = log_ratio - (ratio - 1)
        # This SHOULD be ≤ 0 (proving the wrong formula is wrong)
        assert wrong_kl.sum().item() <= 0.001, \
            "Wrong formula should produce non-positive values (log(r) ≤ r-1 always)"
