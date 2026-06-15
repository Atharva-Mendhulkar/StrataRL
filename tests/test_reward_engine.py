import pytest
import torch
from rewards.structural_reward import structural_reward
from rewards.outcome_verifiers  import verify_numeric, verify_letter, verify_yesno
from rewards.token_repetition   import token_repetition_penalty
from rewards.reward_engine      import score_batch, gdpo_normalize


# ── structural_reward tests ───────────────────────────────────────────────────

class TestStructuralReward:

    def test_valid_gsm8k_completion(self):
        completion = """<think>
<decompose>Janet earns $17/day. Over 5 days that is 5 × 17 dollars.</decompose>
<compute>5 × 17 = 85</compute>
<verify>85 dollars is the total earned over 5 days.</verify>
</think>
<answer>85</answer>"""
        assert structural_reward(completion, domain="gsm8k") == 1.0

    def test_valid_mmlu_completion(self):
        completion = """<think>
<recall>Photosynthesis converts light energy into chemical energy in plants. The primary pigment is chlorophyll.</recall>
<evaluate>The question asks about the primary pigment involved. Chlorophyll is option B.</evaluate>
</think>
<answer>B</answer>"""
        assert structural_reward(completion, domain="mmlu") == 1.0

    def test_valid_strategyqa_completion(self):
        completion = """<think>
<decompose>Does a shark share habitat with a Chihuahua? Sharks are marine, Chihuahuas are domestic land animals.</decompose>
<resolve>Sharks live in oceans. Chihuahuas live in homes/land environments. Their habitats do not overlap.</resolve>
<synthesize>Since they do not share habitat, a shark would not encounter a Chihuahua in the wild.</synthesize>
</think>
<answer>no</answer>"""
        assert structural_reward(completion, domain="strategyqa") == 1.0

    # ── Exploit tests — These now return partial credit (graded rewards) ──────

    def test_blocks_empty_think(self):
        """Reward hacking: <think></think> should fail min-length gate but gets some credit for tags."""
        completion = "<think></think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.35

    def test_blocks_missing_think(self):
        completion = "<answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.25

    def test_blocks_missing_answer(self):
        completion = "<think>Some reasoning here that is long enough to pass gates.</think>"
        assert structural_reward(completion, domain="gsm8k") == 0.175

    def test_blocks_wrong_order(self):
        """answer before think must fail."""
        completion = "<answer>42</answer><think>I thought about it after answering.</think>"
        assert structural_reward(completion, domain="gsm8k") == 0.3

    def test_blocks_garbage_repetition(self):
        """Character repetition garbage must fail."""
        completion = "<think>" + "a" * 200 + "</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.45

    def test_blocks_word_repetition(self):
        """Word-level n-gram repetition must fail."""
        completion = "<think>" + "the answer is correct " * 30 + "</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.45

    def test_blocks_missing_domain_tags_gsm8k(self):
        """GSM8K requires <decompose>, <compute>, <verify> — missing any should fail."""
        completion = """<think>
Janet earns $17/day. Over 5 days that is 85 dollars total.
This is a straightforward multiplication problem and the answer is 85.
</think>
<answer>85</answer>"""
        assert structural_reward(completion, domain="gsm8k") == 0.5

    def test_blocks_missing_domain_tags_strategyqa(self):
        """StrategyQA requires <decompose>, <resolve>, <synthesize>."""
        completion = """<think>
<decompose>Analyzing the question about sharks and Chihuahuas.</decompose>
Sharks live in water. Chihuahuas live on land.
</think>
<answer>no</answer>"""
        # Missing <resolve> and <synthesize>
        assert structural_reward(completion, domain="strategyqa") == 0.5

    def test_think_too_short_for_gsm8k(self):
        """< 80 chars in <think> for gsm8k must fail."""
        completion = "<think>\n<decompose>x</decompose><compute>y</compute><verify>z</verify>\n</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.4


# ── Outcome verifier tests ────────────────────────────────────────────────────

class TestOutcomeVerifiers:

    def test_numeric_exact_match(self):
        assert verify_numeric("<answer>42</answer>", "42") == 1.0

    def test_numeric_fraction_equivalent(self):
        assert verify_numeric("<answer>1/2</answer>", "0.5") == 1.0

    def test_numeric_sympy_simplification(self):
        assert verify_numeric("<answer>2*21</answer>", "42") == 1.0

    def test_numeric_wrong_answer(self):
        assert verify_numeric("<answer>41</answer>", "42") == 0.0

    def test_numeric_no_answer_tag(self):
        assert verify_numeric("The answer is 42", "42") == 0.0

    def test_letter_correct(self):
        assert verify_letter("<answer>B</answer>", "B") == 1.0

    def test_letter_case_insensitive(self):
        assert verify_letter("<answer>b</answer>", "B") == 1.0

    def test_letter_wrong(self):
        assert verify_letter("<answer>A</answer>", "B") == 0.0

    def test_yesno_yes(self):
        assert verify_yesno("<answer>yes</answer>", "yes") == 1.0

    def test_yesno_no(self):
        assert verify_yesno("<answer>no</answer>", "no") == 1.0

    def test_yesno_case_insensitive(self):
        assert verify_yesno("<answer>Yes</answer>", "yes") == 1.0

    def test_yesno_wrong(self):
        assert verify_yesno("<answer>yes</answer>", "no") == 0.0


# ── Token repetition tests ────────────────────────────────────────────────────

class TestTokenRepetition:

    def test_no_repetition(self):
        tokens = list(range(100))   # all unique IDs
        assert token_repetition_penalty(tokens) == 1.0

    def test_excessive_repetition(self):
        pattern = [1, 2, 3, 4, 5]
        tokens  = pattern * 10     # repeats 10 times — exceeds max_reps=4
        assert token_repetition_penalty(tokens) == 0.0

    def test_acceptable_repetition(self):
        pattern = [1, 2, 3, 4, 5]
        tokens  = list(range(50)) + pattern * 3  # repeats 3 times — within threshold
        assert token_repetition_penalty(tokens) == 1.0

    def test_short_completion(self):
        tokens = [1, 2, 3]   # shorter than n=5
        assert token_repetition_penalty(tokens) == 1.0


# ── GDPO normalization tests ──────────────────────────────────────────────────

class TestGDPO:

    def test_znorm_shape_preserved(self):
        x = torch.tensor([[0.0, 1.0, 0.0, 1.0],
                           [1.0, 1.0, 0.0, 0.0]])
        z = gdpo_normalize(x)
        assert z.shape == x.shape

    def test_znorm_row_zero_mean(self):
        x = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        z = gdpo_normalize(x)
        assert abs(z.mean().item()) < 1e-5

    def test_znorm_constant_row_no_nan(self):
        """All same reward in a group → std=0 → should not produce NaN."""
        x = torch.ones(2, 4)
        z = gdpo_normalize(x)
        assert not torch.isnan(z).any()
