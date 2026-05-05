import pytest
import torch
import torch.nn.functional as F
from m4.m4_rollout_engine import build_m4_engine


class TestLogprobAlignment:

    @pytest.fixture(scope="class")
    def engine(self):
        return build_m4_engine("Qwen/Qwen2.5-0.5B-Instruct")

    def test_logprob_length_matches_token_ids(self, engine):
        """Most critical alignment test: len(logprobs) == len(token_ids) for every completion."""
        rollouts = engine.generate(
            prompts=["What is 2 + 2?"],
            G=3,
            max_new_tokens=50,
        )
        for rollout in rollouts:
            for token_ids, logprobs in zip(rollout["token_ids"], rollout["rollout_logprobs"]):
                assert len(logprobs) == len(token_ids), (
                    f"Alignment mismatch: {len(logprobs)} logprobs vs {len(token_ids)} tokens"
                )

    def test_logprobs_are_finite(self, engine):
        """Logprobs must be finite (no -inf, no NaN)."""
        rollouts = engine.generate(
            prompts=["What is 3 × 4?"],
            G=2,
            max_new_tokens=30,
        )
        for rollout in rollouts:
            for logprobs in rollout["rollout_logprobs"]:
                for lp in logprobs:
                    assert torch.isfinite(torch.tensor(lp)), f"Non-finite logprob: {lp}"

    def test_logprobs_are_negative(self, engine):
        """Log probabilities of individual tokens must be ≤ 0."""
        rollouts = engine.generate(
            prompts=["Hello"],
            G=2,
            max_new_tokens=20,
        )
        for rollout in rollouts:
            for logprobs in rollout["rollout_logprobs"]:
                for lp in logprobs:
                    assert lp <= 0.01, f"Logprob > 0: {lp} — this would be a probability > 1"

    def test_g_completions_are_diverse(self, engine):
        """G rollouts with temperature > 0 should not all be identical."""
        rollouts = engine.generate(
            prompts=["Solve: 15 × 7 = ?"],
            G=4,
            temperature=0.9,
            max_new_tokens=30,
        )
        texts = set(rollouts[0]["completions"])
        assert len(texts) > 1, (
            "All G completions are identical — temperature sampling may be broken. "
            "Diversity in rollouts is essential for non-zero GRPO advantages."
        )
