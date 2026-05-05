# build.md — StrataRL M4 Validation Runbook
## MacBook Air M4 24GB | Pre-Kaggle Architecture Validation

---

## PURPOSE OF THIS FILE

This runbook validates the StrataRL architecture on M4 before committing to Kaggle P100
training runs. The goal is NOT to train a production model. The goal is to:

1. Catch silent bugs (KL sign errors, alignment mismatches, reward hacking exploits)
2. Validate every component in isolation before integration
3. Run a 50-step smoke test that exercises the full pipeline
4. Confirm reward engine correctness on known examples
5. Confirm SAN math produces the expected gradient directions
6. Identify any M4-specific issues before they surface on Kaggle

**Time budget:** ~4 hours total (component tests ~2hr, smoke test ~2hr)
**Model for testing:** Qwen2.5-0.5B-Instruct (fast iteration; architecture is identical to 3B)
**Do NOT use vLLM on M4** — vLLM requires CUDA. Use HuggingFace generate() with MPS backend.
**Do NOT run SFT warmup on M4** — too slow. Use base instruct checkpoint directly.

---

## M4 STACK DIFFERENCES FROM KAGGLE

| Component | Kaggle P100 | M4 Local | Action |
|-----------|-------------|----------|--------|
| Inference engine | vLLM (PagedAttention) | HF generate() + MPS | Replace rollout_engine.py with m4_rollout_engine.py |
| Training backend | CUDA + Unsloth kernels | PyTorch MPS | Replace unsloth calls with standard HF Trainer |
| Quantization | 4-bit NF4 (bitsandbytes) | bfloat16 full precision | bitsandbytes not supported on MPS |
| LoRA library | Unsloth LoRA | PEFT LoRA standard | Drop-in replacement |
| Multi-GPU | DataParallel | Single device (MPS) | Remove all distributed code paths |
| Logprobs source | vLLM SamplingParams logprobs=1 | HF generate() scores | Different extraction method — test carefully |
| Model size | Qwen2.5-3B (primary) | Qwen2.5-0.5B (testing) | Architecture identical, weights smaller |

**Critical:** Every component except the model size and device backend is identical.
All bugs found on M4 will be present on Kaggle. All fixes apply directly.

---

## ENVIRONMENT SETUP

### [x] Step 0.1 — Create isolated environment

```bash
cd ~/projects
mkdir stratarl && cd stratarl

python3 -m venv .venv
source .venv/bin/activate

# Verify Python 3.11+
python --version
```

### [x] Step 0.2 — Install dependencies (M4-specific, no CUDA packages)

```bash
# Core ML stack — MPS-compatible versions
pip install torch torchvision torchaudio   # auto-detects Apple Silicon
pip install transformers==4.47.0
pip install datasets==3.1.0
pip install peft==0.14.0
pip install trl==0.12.0                    # GRPO implementation
pip install accelerate==1.2.0

# Math + verification
pip install sympy==1.13.3
pip install scipy numpy

# Monitoring
pip install wandb
pip install rich                           # pretty terminal output

# Testing
pip install pytest pytest-cov

# Explicitly NOT installing (CUDA-only):
# bitsandbytes — not supported on MPS
# vllm — requires CUDA
# unsloth — requires CUDA + triton
```

### [x] Step 0.3 — Verify MPS device is available

```python
# run: python -c "import torch; print(torch.backends.mps.is_available())"
# Expected output: True
# If False: reinstall torch with: pip install --upgrade torch
```

### [x] Step 0.4 — Download test model

```python
# Run this once — downloads ~1GB to ~/.cache/huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
# Expected: ~494M parameters
```

### [x] Step 0.5 — Verify W&B (optional for M4 testing, recommended)

```bash
wandb login
# Use project: stratarl_m4_validation
```

---

## REPOSITORY STRUCTURE (M4 VERSION)

```
stratarl/
├── CLAUDE.md                          ← this file
│
├── m4/                                ← M4-specific overrides (not used on Kaggle)
│   ├── m4_rollout_engine.py           ← HF generate() replaces vLLM
│   ├── m4_train.py                    ← MPS training loop (no Unsloth)
│   └── m4_config.yaml                ← M4 smoke-test config
│
├── rewards/
│   ├── reward_engine.py
│   ├── outcome_verifiers.py
│   ├── structural_reward.py
│   └── token_repetition.py
│
├── training/
│   ├── advantage.py                   ← SAN + normalization (device-agnostic)
│   ├── policy_update.py               ← GRPO loss (device-agnostic)
│   └── fallback.py
│
├── curriculum/
│   ├── ucb_scheduler.py
│   └── collapse_detector.py
│
├── monitoring/
│   └── monitor.py
│
├── data/
│   ├── templates.py
│   └── loaders.py
│
├── tests/
│   ├── test_reward_engine.py          ← unit tests for all reward components
│   ├── test_san_advantage.py          ← unit tests for SAN math
│   ├── test_kl_computation.py         ← unit tests for KL correctness
│   ├── test_alignment.py              ← logprob alignment validation
│   ├── test_structural_reward.py      ← reward hacking exploit tests
│   └── test_ucb_scheduler.py          ← curriculum scheduling tests
│
└── scripts/
    ├── run_component_tests.sh
    └── run_smoke_test.sh
```

---

## PHASE 1 — COMPONENT IMPLEMENTATION

Build each component in isolation. Test it before moving on.
Do NOT integrate until each component passes its unit tests.

---

### COMPONENT 1: Reward Engine

**File:** `rewards/structural_reward.py`

#### [x] 1.1 — Implement domain templates

```python
# rewards/structural_reward.py

import re
from collections import Counter
from typing import List

# ── Domain template definitions ──────────────────────────────────────────────

DOMAIN_TEMPLATES = {
    "gsm8k":       ["decompose", "compute", "verify"],
    "mmlu":        ["recall", "evaluate"],
    "strategyqa":  ["decompose", "resolve", "synthesize"],
    "aqua_rat":    ["decompose", "compute", "verify"],
}

DOMAIN_MIN_THINK_CHARS = {
    "gsm8k":       80,
    "mmlu":        100,
    "strategyqa":  100,
    "aqua_rat":    120,
    "default":     80,
}

UNIVERSAL_MIN_THINK_CHARS = 50
MAX_NGRAM_REPEAT          = 4

# ── Compiled patterns ─────────────────────────────────────────────────────────

THINK_PATTERN  = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

GARBAGE_PATTERNS = [
    re.compile(r"(.)\1{20,}",                  re.DOTALL),  # 20+ repeated chars
    re.compile(r"(\b\w+\b)(\s+\1){5,}",        re.DOTALL),  # same word 5+ times
    re.compile(r"^[\s\W]*$",                   re.DOTALL),  # whitespace/punct only
]

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_repetition_score(text: str, n: int = 5) -> int:
    words  = text.lower().split()
    if len(words) < n:
        return 0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return max(Counter(ngrams).values()) if ngrams else 0


def structural_reward(completion: str, domain: str = "default") -> float:
    """
    Returns 1.0 iff all gates pass. 0.0 on any failure.
    
    Gates (in order):
      1. Both <think> and <answer> present, correctly ordered
      2. <think> content >= domain-appropriate minimum length
      3. <answer> content non-empty
      4. No garbage patterns in <think>
      5. Word-level n-gram repetition <= MAX_NGRAM_REPEAT
      6. All domain-specific template tags present in <think>
    """
    think_m  = THINK_PATTERN.search(completion)
    answer_m = ANSWER_PATTERN.search(completion)

    # Gate 1
    if not think_m or not answer_m:
        return 0.0
    if think_m.end() > answer_m.start():
        return 0.0

    think   = think_m.group(1).strip()
    answer  = answer_m.group(1).strip()

    # Gate 2
    min_chars = max(
        UNIVERSAL_MIN_THINK_CHARS,
        DOMAIN_MIN_THINK_CHARS.get(domain, DOMAIN_MIN_THINK_CHARS["default"])
    )
    if len(think) < min_chars:
        return 0.0

    # Gate 3
    if not answer:
        return 0.0

    # Gate 4
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(think):
            return 0.0

    # Gate 5
    if compute_repetition_score(think, n=5) > MAX_NGRAM_REPEAT:
        return 0.0

    # Gate 6
    required_tags = DOMAIN_TEMPLATES.get(domain, [])
    for tag in required_tags:
        if f"<{tag}>" not in think or f"</{tag}>" not in think:
            return 0.0

    return 1.0
```

#### [x] 1.2 — Implement token repetition penalty

```python
# rewards/token_repetition.py

from collections import Counter
from typing import List

TOKEN_NGRAM_SIZE     = 5
TOKEN_NGRAM_MAX_REPS = 4

def token_repetition_penalty(
    completion_token_ids: List[int],
    n: int = TOKEN_NGRAM_SIZE,
    max_reps: int = TOKEN_NGRAM_MAX_REPS,
) -> float:
    if len(completion_token_ids) < n:
        return 1.0
    ngrams  = [tuple(completion_token_ids[i:i+n]) for i in range(len(completion_token_ids)-n+1)]
    if not ngrams:
        return 1.0
    return 0.0 if max(Counter(ngrams).values()) > max_reps else 1.0
```

#### [x] 1.3 — Implement outcome verifiers

```python
# rewards/outcome_verifiers.py

import re
import sympy

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def extract_answer(completion: str) -> str | None:
    m = ANSWER_PATTERN.search(completion)
    return m.group(1).strip() if m else None


def normalize_number_str(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return str(float(s))
    except:
        return s


def verify_numeric(completion: str, ground_truth: str) -> float:
    """GSM8K / AQuA-RAT: symbolic numeric equivalence via SymPy."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    try:
        result = sympy.simplify(sympy.sympify(answer) - sympy.sympify(ground_truth))
        return 1.0 if result == 0 else 0.0
    except:
        return float(normalize_number_str(answer) == normalize_number_str(ground_truth))


def verify_letter(completion: str, ground_truth: str) -> float:
    """MMLU: exact letter match A/B/C/D."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    m = re.search(r'\b([A-Da-d])\b', answer.strip())
    return float(bool(m) and m.group(1).upper() == ground_truth.strip().upper())


def verify_yesno(completion: str, ground_truth: str) -> float:
    """StrategyQA: yes/no match."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    return float(answer.strip().lower() == ground_truth.strip().lower())


DOMAIN_VERIFIERS = {
    "gsm8k":      verify_numeric,
    "aqua_rat":   verify_numeric,
    "mmlu":       verify_letter,
    "strategyqa": verify_yesno,
}
```

#### [x] 1.4 — Implement GDPO reward aggregation

```python
# rewards/reward_engine.py

import torch
from typing import List, Dict
from rewards.structural_reward import structural_reward
from rewards.token_repetition  import token_repetition_penalty
from rewards.outcome_verifiers import DOMAIN_VERIFIERS


def gdpo_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-normalize a [B, G] reward tensor across G per row."""
    mu  = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-8
    return (x - mu) / std


def score_batch(
    rollouts:      List[Dict],
    ground_truths: List[str],
    domains:       List[str],
    w_outcome:     float = 0.7,
    w_struct:      float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score a batch of rollouts.
    
    Returns:
        gdpo_rewards: [B, G]  — GDPO-aggregated composite reward
        raw_rewards:  [3, B, G] — [outcome, struct, token_rep] before aggregation
    """
    B = len(rollouts)
    G = len(rollouts[0]["completions"])

    outcome_r = torch.zeros(B, G)
    struct_r  = torch.zeros(B, G)
    token_rep = torch.ones(B, G)

    for i, (rollout, gt, domain) in enumerate(zip(rollouts, ground_truths, domains)):
        verifier = DOMAIN_VERIFIERS.get(domain, DOMAIN_VERIFIERS["gsm8k"])
        for j, completion in enumerate(rollout["completions"]):
            outcome_r[i, j] = verifier(completion, gt)
            struct_r[i, j]  = structural_reward(completion, domain=domain)
            token_rep[i, j] = token_repetition_penalty(
                rollout["token_ids"][j]
            )

    # Gate structural with token repetition
    struct_gated = struct_r * token_rep

    # GDPO: normalize each signal independently, then combine
    z_outcome = gdpo_normalize(outcome_r)
    z_struct  = gdpo_normalize(struct_gated)

    gdpo_rewards = w_outcome * z_outcome + w_struct * z_struct
    raw_rewards  = torch.stack([outcome_r, struct_r, token_rep], dim=0)

    return gdpo_rewards, raw_rewards
```

#### [x] 1.5 — Write and run reward engine tests

```python
# tests/test_reward_engine.py

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

    # ── Exploit tests — these MUST return 0.0 ──────────────────────────────

    def test_blocks_empty_think(self):
        """Reward hacking: <think></think> should fail min-length gate."""
        completion = "<think></think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_missing_think(self):
        completion = "<answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_missing_answer(self):
        completion = "<think>Some reasoning here that is long enough to pass gates.</think>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_wrong_order(self):
        """answer before think must fail."""
        completion = "<answer>42</answer><think>I thought about it after answering.</think>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_garbage_repetition(self):
        """Character repetition garbage must fail."""
        completion = "<think>" + "a" * 200 + "</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_word_repetition(self):
        """Word-level n-gram repetition must fail."""
        completion = "<think>" + "the answer is correct " * 30 + "</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_missing_domain_tags_gsm8k(self):
        """GSM8K requires <decompose>, <compute>, <verify> — missing any should fail."""
        completion = """<think>
Janet earns $17/day. Over 5 days that is 85 dollars total.
This is a straightforward multiplication problem and the answer is 85.
</think>
<answer>85</answer>"""
        assert structural_reward(completion, domain="gsm8k") == 0.0

    def test_blocks_missing_domain_tags_strategyqa(self):
        """StrategyQA requires <decompose>, <resolve>, <synthesize>."""
        completion = """<think>
<decompose>Analyzing the question about sharks and Chihuahuas.</decompose>
Sharks live in water. Chihuahuas live on land.
</think>
<answer>no</answer>"""
        # Missing <resolve> and <synthesize>
        assert structural_reward(completion, domain="strategyqa") == 0.0

    def test_think_too_short_for_gsm8k(self):
        """< 80 chars in <think> for gsm8k must fail."""
        completion = "<think>\n<decompose>x</decompose><compute>y</compute><verify>z</verify>\n</think><answer>42</answer>"
        assert structural_reward(completion, domain="gsm8k") == 0.0


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


# ── Run instructions ──────────────────────────────────────────────────────────
# pytest tests/test_reward_engine.py -v
# Expected: all 25+ tests PASS
# Any failure here = critical bug that will corrupt training on Kaggle
```

**Run:**
```bash
pytest tests/test_reward_engine.py -v
```

**Expected:** All tests pass. Zero failures, zero errors.
**If any test fails:** Fix the component before proceeding. Do not move to Component 2.

---

### COMPONENT 2: SAN Advantage Engine

**File:** `training/advantage.py`

#### [x] 2.1 — Implement SAN

```python
# training/advantage.py

import torch
import numpy as np
from typing import List, Dict

LENGTH_NORM_CLAMP = 512   # completions longer than this get equal treatment
ADVANTAGE_CLIP    = 5.0


def compute_san_advantages(
    rewards:    torch.Tensor,   # [B, G] — GDPO-aggregated composite rewards
    domains:    List[str],      # length B — domain label per prompt
    eps:        float = 1e-8,
) -> torch.Tensor:
    """
    Stratified Advantage Normalization.

    Computes Z-score normalization WITHIN each domain stratum independently.
    Prevents cross-stratum bias from heterogeneous reward distributions.

    Returns advantages: [B, G], clipped to ±ADVANTAGE_CLIP
    """
    advantages = torch.zeros_like(rewards)
    unique_domains = set(domains)

    for domain in unique_domains:
        stratum_idx = [i for i, d in enumerate(domains) if d == domain]

        if len(stratum_idx) < 2:
            # Single-sample stratum: zero advantage (no comparison possible)
            advantages[stratum_idx] = 0.0
            continue

        stratum_rewards = rewards[stratum_idx]       # [n_d, G]
        flat            = stratum_rewards.flatten()

        mu    = flat.mean()
        sigma = flat.std() + eps

        normalized               = (stratum_rewards - mu) / sigma
        advantages[stratum_idx]  = torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    return advantages


def expand_advantages_to_tokens(
    advantages:          torch.Tensor,        # [B, G]
    completion_lengths:  List[List[int]],     # [B][G] — token counts per completion
    use_length_norm:     bool = True,
) -> torch.Tensor:
    """
    Expand scalar per-completion advantages to per-token advantages.
    Applies length normalization clamped at LENGTH_NORM_CLAMP tokens.
    """
    token_advantages = []
    for b in range(advantages.shape[0]):
        for g in range(advantages.shape[1]):
            adv_scalar = advantages[b, g].item()
            length     = completion_lengths[b][g]

            if use_length_norm and length > 0:
                clamped_length = min(length, LENGTH_NORM_CLAMP)
                norm_factor    = 1.0 / (clamped_length ** 0.5)
                adv_scalar     = adv_scalar * norm_factor

            token_advantages.extend([adv_scalar] * length)

    return torch.tensor(token_advantages, dtype=torch.float32)
```

#### [x] 2.2 — Write and run SAN tests

```python
# tests/test_san_advantage.py

import pytest
import torch
from training.advantage import compute_san_advantages, expand_advantages_to_tokens


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
        assert adv_100 > adv_400, \
            "Shorter completion should have higher per-token advantage (1/sqrt(100) > 1/sqrt(400))"


def _global_normalize(rewards: torch.Tensor) -> torch.Tensor:
    """Reference implementation of standard (broken) global GRPO normalization."""
    flat  = rewards.flatten()
    mu    = flat.mean()
    sigma = flat.std() + 1e-8
    return (rewards - mu) / sigma


# Run: pytest tests/test_san_advantage.py -v
# The test_multi_domain_independent_normalization test is the most critical.
# It directly proves SAN solves the problem that motivates StrataRL.
```

**Run:**
```bash
pytest tests/test_san_advantage.py -v
```

**Expected:** All tests pass. The `test_multi_domain_independent_normalization` test is the architectural proof-of-concept. If it fails, the core hypothesis of StrataRL is not implemented correctly.

---

### COMPONENT 3: KL Computation & Policy Update

**File:** `training/policy_update.py`

#### [x] 3.1 — Implement GRPO loss with correct KL formula

```python
# training/policy_update.py

import torch
import torch.nn.functional as F
from typing import Dict


def grpo_loss(
    policy_model,
    input_ids:        torch.Tensor,    # [B*G, seq_len]
    attention_mask:   torch.Tensor,    # [B*G, seq_len]
    completion_mask:  torch.Tensor,    # [B*G, seq_len] — 1 on completion tokens
    advantages:       torch.Tensor,    # [B*G, seq_len] — token-level advantages
    old_logprobs:     torch.Tensor,    # [B*G, seq_len] — logprobs from rollout
    beta:             float = 0.01,
    clip_eps:         float = 0.2,
    entropy_floor:    float = 0.0,     # > 0 only for EXP_04 (β=0)
    entropy_coeff:    float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    GRPO policy update loss.

    Key design decisions:
    - No separate reference model: old_logprobs captured at rollout time from HF generate
    - KL formula: p_old × (log_p_old - log_p_new) — always non-negative in expectation
    - Entropy floor regularization: activates only when entropy_floor > 0 (EXP_04)
    """
    device = input_ids.device

    # Forward pass through policy
    outputs      = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits       = outputs.logits[:, :-1, :]                    # [B*G, seq-1, vocab]

    labels       = input_ids[:, 1:]                             # [B*G, seq-1]
    log_probs    = F.log_softmax(logits, dim=-1)
    policy_logp  = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B*G, seq-1]

    # Align old_logprobs to shifted label positions
    old_logp_aligned = old_logprobs[:, 1:].detach()             # [B*G, seq-1]

    # ── Alignment assertion — NEVER REMOVE ────────────────────────────────
    assert old_logp_aligned.shape == policy_logp.shape, (
        f"CRITICAL ALIGNMENT FAILURE: "
        f"old_logp_aligned {old_logp_aligned.shape} != "
        f"policy_logp {policy_logp.shape}. "
        f"Gradient computation is silently wrong. Halting immediately."
    )

    # ── Policy gradient (PPO-style surrogate) ─────────────────────────────
    log_ratio = policy_logp - old_logp_aligned
    ratio     = torch.exp(log_ratio)

    comp_mask  = completion_mask[:, 1:]                         # [B*G, seq-1]
    adv_tokens = advantages[:, 1:]                              # [B*G, seq-1]

    surr1 = ratio * adv_tokens
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tokens
    policy_loss = -torch.min(surr1, surr2)

    # ── KL divergence: CORRECT formula ────────────────────────────────────
    # = p_old(t) × log(p_old(t) / p_new(t))
    # = exp(old_logp) × (old_logp - policy_logp)
    # Non-negative in expectation; individual tokens may be negative (acceptable)
    # Does NOT assume ratio ≈ 1 (valid for large deviations, critical for β=0)
    #
    # WRONG formula (do not use):
    #   kl = log_ratio - (ratio - 1)   ← always ≤ 0, REWARDS divergence
    kl_per_token = torch.exp(old_logp_aligned) * (old_logp_aligned - policy_logp)

    # ── Entropy ───────────────────────────────────────────────────────────
    entropy = -(torch.exp(policy_logp) * policy_logp * comp_mask).sum() / (comp_mask.sum() + 1e-8)

    # ── Loss aggregation ──────────────────────────────────────────────────
    policy_loss_mean = (policy_loss * comp_mask).sum() / (comp_mask.sum() + 1e-8)
    kl_mean          = (kl_per_token * comp_mask).sum() / (comp_mask.sum() + 1e-8)

    total_loss = policy_loss_mean + beta * kl_mean

    # Entropy floor regularization (EXP_04 only: β=0, prevents collapse)
    entropy_deficit = torch.tensor(0.0, device=device)
    if entropy_floor > 0.0:
        entropy_deficit = F.relu(torch.tensor(entropy_floor, device=device) - entropy)
        total_loss      = total_loss - entropy_coeff * entropy_deficit

    return {
        "loss":             total_loss,
        "policy_loss":      policy_loss_mean,
        "kl":               kl_mean,
        "entropy":          entropy,
        "entropy_deficit":  entropy_deficit,
        "ratio_mean":       (ratio * comp_mask).sum() / (comp_mask.sum() + 1e-8),
        "ratio_max":        ratio.max(),
    }
```

#### [x] 3.2 — Write and run KL tests

```python
# tests/test_kl_computation.py

import pytest
import torch
import torch.nn.functional as F
from training.policy_update import grpo_loss


class TestKLFormula:

    def test_kl_is_nonnegative_in_expectation(self):
        """
        The KL formula p_old × (log_p_old - log_p_new) should be
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
        kl = torch.exp(old_token_logp) * (old_token_logp - new_token_logp)
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
        kl   = torch.exp(logp) * (logp - logp)
        assert abs(kl.sum().item()) < 1e-6

    def test_kl_positive_for_diverged_policy(self):
        """If new policy has moved away from old, KL > 0."""
        old_logp = torch.tensor([[-1.0, -2.0, -3.0, -0.5]])
        new_logp = torch.tensor([[-0.5, -3.0, -2.0, -1.5]])   # different distribution
        kl = torch.exp(old_logp) * (old_logp - new_logp)
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


# Run: pytest tests/test_kl_computation.py -v
```

**Run:**
```bash
pytest tests/test_kl_computation.py -v
```

**Critical test:** `test_kl_is_nonnegative_in_expectation`. If this fails, the KL formula has the sign error that would cause training to reward divergence rather than penalize it.

---

### COMPONENT 4: M4 Rollout Engine

**File:** `m4/m4_rollout_engine.py`

#### [x] 4.1 — Implement HF generate() rollout with logprob extraction

```python
# m4/m4_rollout_engine.py
"""
M4-specific rollout engine.
Replaces vLLM with HuggingFace generate() + MPS backend.
Captures per-token logprobs for use in GRPO loss (replaces reference model).

KAGGLE NOTE: This file is NOT used on Kaggle. On Kaggle, use rollout/rollout_engine.py
which uses vLLM with PagedAttention. The logprob extraction logic below is
the equivalent of vLLM's logprobs=1 parameter.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class M4RolloutEngine:

    def __init__(self, model, tokenizer, device="mps"):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def generate(
        self,
        prompts:        List[str],
        G:              int   = 4,       # M4: use G=4, not G=8 (memory)
        temperature:    float = 0.85,
        top_p:          float = 0.95,
        max_new_tokens: int   = 512,     # M4: shorter than Kaggle (768→512)
        min_new_tokens: int   = 50,
    ) -> List[Dict]:
        """
        Generate G completions per prompt and capture per-token logprobs.
        
        Returns list of dicts with keys:
          - prompt:           str
          - completions:      List[str]          (G completions)
          - token_ids:        List[List[int]]    (completion tokens only)
          - rollout_logprobs: List[List[float]]  (logprob of each chosen token)
          - finish_reasons:   List[str]
        """
        rollouts = []

        for prompt in prompts:
            prompt_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self.device)
            prompt_len = prompt_ids.shape[1]

            completions      = []
            all_token_ids    = []
            all_logprobs     = []
            finish_reasons   = []

            for _ in range(G):
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens    = max_new_tokens,
                    min_new_tokens    = min_new_tokens,
                    do_sample         = True,
                    temperature       = temperature,
                    top_p             = top_p,
                    repetition_penalty= 1.1,
                    return_dict_in_generate = True,
                    output_scores     = True,           # ← enables logprob extraction
                    pad_token_id      = self.tokenizer.eos_token_id,
                    eos_token_id      = self.tokenizer.eos_token_id,
                )

                # Extract completion token IDs (strip prompt)
                full_ids     = outputs.sequences[0]
                comp_ids     = full_ids[prompt_len:].tolist()

                # Extract per-token logprobs from output.scores
                # outputs.scores: tuple of [vocab_size] tensors, one per generated token
                token_logps  = []
                for step_idx, step_scores in enumerate(outputs.scores):
                    if step_idx >= len(comp_ids):
                        break
                    log_probs_step = F.log_softmax(step_scores[0], dim=-1)
                    chosen_token   = comp_ids[step_idx]
                    token_logps.append(log_probs_step[chosen_token].item())

                # ── Alignment assertion ───────────────────────────────────────
                assert len(token_logps) == len(comp_ids), (
                    f"ALIGNMENT ERROR: token_logps ({len(token_logps)}) != "
                    f"comp_ids ({len(comp_ids)}). "
                    f"This will corrupt GRPO gradient computation."
                )

                completion_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                finish_reason   = "stop" if comp_ids[-1] == self.tokenizer.eos_token_id else "length"

                completions.append(completion_text)
                all_token_ids.append(comp_ids)
                all_logprobs.append(token_logps)
                finish_reasons.append(finish_reason)

            rollouts.append({
                "prompt":           prompt,
                "completions":      completions,
                "token_ids":        all_token_ids,
                "rollout_logprobs": all_logprobs,
                "finish_reasons":   finish_reasons,
            })

        return rollouts


def build_m4_engine(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> M4RolloutEngine:
    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = torch.bfloat16,
    )
    return M4RolloutEngine(model, tokenizer, device=device)
```

#### [x] 4.2 — Test logprob extraction correctness

```python
# tests/test_alignment.py

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


# Run: pytest tests/test_alignment.py -v
# Note: this test requires the model to be downloaded (~1GB)
```

**Run:**
```bash
pytest tests/test_alignment.py -v
```

---

### COMPONENT 5: UCB Curriculum Scheduler

#### [x] 5.1 — Implement UCB scheduler

```python
# curriculum/ucb_scheduler.py

import numpy as np
from typing import Dict, List


class UCBCurriculumScheduler:
    """
    Multi-armed bandit curriculum scheduler.
    Dynamically allocates training compute to domains where the model is actively learning.
    
    Exploitation term: Ā_d = moving average of |advantage| (how much model is improving)
    Exploration term:  c × sqrt(ln(N) / n_d) (ensures all domains visited)
    """

    def __init__(
        self,
        domains:  List[str],
        c:        float = 0.5,     # exploration coefficient
        alpha:    float = 0.1,     # EMA smoothing factor
    ):
        self.domains      = domains
        self.c            = c
        self.alpha        = alpha
        self.total_steps  = 0

        self.domain_avg_advantage    = {d: 0.5  for d in domains}
        self.domain_sample_counts    = {d: 1    for d in domains}
        self.domain_collapse_types   = {d: "HEALTHY" for d in domains}

    def update(self, domain_advantages: Dict[str, List[float]]):
        self.total_steps += 1

        for domain, advantages in domain_advantages.items():
            if not advantages:
                continue
            advs         = np.array(advantages)
            avg_abs_adv  = np.mean(np.abs(advs))
            pos_ratio    = np.mean(advs > 0)
            # Directional health: deviation from 0.5 means saturation or failure
            dir_quality  = 1.0 - abs(pos_ratio - 0.5) * 2
            combined     = 0.7 * avg_abs_adv + 0.3 * dir_quality

            self.domain_avg_advantage[domain] = (
                (1 - self.alpha) * self.domain_avg_advantage[domain]
                + self.alpha * combined
            )
            self.domain_sample_counts[domain] += len(advantages)

    def set_collapse_types(self, collapse_types: Dict[str, str]):
        self.domain_collapse_types = collapse_types

    def get_weights(self) -> Dict[str, float]:
        N = self.total_steps + 1
        raw_scores = {}

        for d in self.domains:
            exploitation = self.domain_avg_advantage[d]
            exploration  = self.c * np.sqrt(np.log(N) / self.domain_sample_counts[d])
            ucb_score    = exploitation + exploration

            collapse = self.domain_collapse_types.get(d, "HEALTHY")
            if collapse == "ALL_CORRECT":
                raw_scores[d] = float('-inf')     # fully saturated — reallocate
            elif collapse == "ALL_WRONG":
                raw_scores[d] = max(ucb_score * 0.3, -3.0)   # suppress but don't eliminate
            else:
                raw_scores[d] = ucb_score

        # Numerically stable softmax (handles -inf correctly)
        vals = np.array(list(raw_scores.values()), dtype=np.float64)

        if np.all(np.isinf(vals) & (vals < 0)):
            # Emergency: all domains collapsed — reset
            for d in self.domains:
                self.domain_avg_advantage[d]  = 0.5
                self.domain_sample_counts[d]  = 1
            vals = np.ones(len(self.domains))

        finite_max   = np.nanmax(vals[~np.isinf(vals)]) if not np.all(np.isinf(vals)) else 0
        vals_shifted = vals - finite_max
        exp_vals     = np.exp(np.clip(vals_shifted, -700, 0))
        exp_vals[np.isinf(vals) & (vals < 0)] = 0.0
        total        = exp_vals.sum()

        if total < 1e-10:
            normalized = np.ones(len(self.domains)) / len(self.domains)
        else:
            normalized = exp_vals / total

        return {d: float(normalized[i]) for i, d in enumerate(self.domains)}

    def sample_domain(self) -> str:
        weights = self.get_weights()
        domains = list(weights.keys())
        probs   = [weights[d] for d in domains]
        return np.random.choice(domains, p=probs)
```

#### [x] 5.2 — Write and run scheduler tests

```python
# tests/test_ucb_scheduler.py

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


# Run: pytest tests/test_ucb_scheduler.py -v
```

**Run:**
```bash
pytest tests/test_ucb_scheduler.py -v
```

---

## PHASE 2 — INTEGRATION TESTS

Run these after ALL component tests pass.

#### [ ] 6.1 — Run full test suite

```bash
pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

**Expected:** 0 failures, 0 errors across all test files.
**If any fail:** Fix before proceeding to smoke test.

#### [ ] 6.2 — Integration: reward engine end-to-end

```python
# tests/test_integration_reward.py
"""
Tests the full reward pipeline: rollout → reward engine → SAN → advantages
using realistic (not synthetic) completions from the 0.5B model.
"""

import torch
from m4.m4_rollout_engine import build_m4_engine
from rewards.reward_engine  import score_batch
from training.advantage     import compute_san_advantages

GSMS_PROMPTS = [
    ("A store sold 15 apples at $2 each and 8 oranges at $3 each. How much total?",
     "54", "gsm8k"),
    ("Janet saves $17 per week. How much does she save in 6 weeks?",
     "102", "gsm8k"),
]

STRAT_PROMPTS = [
    ("Would a Venus flytrap survive in the Arctic?", "no", "strategyqa"),
]

def test_full_reward_pipeline():
    engine = build_m4_engine()

    all_prompts = [p for p, _, _ in GSMS_PROMPTS + STRAT_PROMPTS]
    all_gts     = [g for _, g, _ in GSMS_PROMPTS + STRAT_PROMPTS]
    all_domains = [d for _, _, d in GSMS_PROMPTS + STRAT_PROMPTS]

    # Format prompts with domain templates
    formatted = [_format_prompt(p, d) for p, d in zip(all_prompts, all_domains)]

    rollouts = engine.generate(formatted, G=4, max_new_tokens=200)

    gdpo_rewards, raw_rewards = score_batch(rollouts, all_gts, all_domains)

    # Shape checks
    B, G = len(rollouts), 4
    assert gdpo_rewards.shape == (B, G)
    assert raw_rewards.shape  == (3, B, G)

    # Value range checks
    assert not torch.isnan(gdpo_rewards).any(), "NaN in GDPO rewards"
    assert not torch.isinf(gdpo_rewards).any(), "Inf in GDPO rewards"

    # SAN advantages
    advantages = compute_san_advantages(gdpo_rewards, all_domains)
    assert advantages.shape == (B, G)
    assert advantages.max().item() <= 5.01
    assert advantages.min().item() >= -5.01
    assert not torch.isnan(advantages).any()

    print("\n✓ Full reward pipeline integration test passed")
    print(f"  Mean outcome reward: {raw_rewards[0].mean():.3f}")
    print(f"  Mean structural reward: {raw_rewards[1].mean():.3f}")
    print(f"  Mean GDPO reward: {gdpo_rewards.mean():.3f}")


def _format_prompt(question: str, domain: str) -> str:
    template_instructions = {
        "gsm8k": "Use <decompose>, <compute>, <verify> tags inside <think>.",
        "strategyqa": "Use <decompose>, <resolve>, <synthesize> tags inside <think>.",
        "mmlu": "Use <recall>, <evaluate> tags inside <think>.",
    }
    inst = template_instructions.get(domain, "Show your reasoning.")
    return (
        f"<|system|>\nYou are a careful reasoning assistant. {inst}\n"
        f"Format: <think>...</think><answer>...</answer>\n"
        f"<|user|>\n{question}\n<|assistant|>\n"
    )


if __name__ == "__main__":
    test_full_reward_pipeline()
```

**Run:**
```bash
python tests/test_integration_reward.py
```

---

## PHASE 3 — SMOKE TEST (50 STEPS)

#### [ ] 7.1 — M4 smoke test config

```yaml
# m4/m4_config.yaml

model_id:       "Qwen/Qwen2.5-0.5B-Instruct"
device:         "mps"
dtype:          "bfloat16"

# Training
num_steps:      50
G:              4
batch_size:     2

# LoRA (PEFT, not Unsloth)
lora_r:         8
lora_alpha:     16
target_modules: ["q_proj", "v_proj"]

# GRPO
beta:           0.01
clip_eps:       0.2
lr:             1e-5
max_new_tokens: 200
min_new_tokens: 30

# Rewards
w_outcome:      0.7
w_struct:       0.3

# Domains (reduced for speed)
domains:        ["gsm8k", "strategyqa"]
samples_per_domain: 20

# Monitoring
log_every:      5
eval_every:     25
wandb_project:  "stratarl_m4_smoke"
```

#### [ ] 7.2 — Smoke test training script

```python
# m4/m4_train.py

import torch
import wandb
import yaml
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from m4.m4_rollout_engine      import M4RolloutEngine
from rewards.reward_engine      import score_batch
from training.advantage         import compute_san_advantages, expand_advantages_to_tokens
from training.policy_update     import grpo_loss
from curriculum.ucb_scheduler   import UCBCurriculumScheduler
from data.loaders               import load_domain_samples
from monitoring.monitor         import SmokeTestMonitor


def run_smoke_test(config_path: str = "m4/m4_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device    = cfg["device"] if torch.backends.mps.is_available() else "cpu"
    print(f"[StrataRL M4] Device: {device}")

    # ── Model setup ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = cfg["lora_r"],
        lora_alpha     = cfg["lora_alpha"],
        target_modules = cfg["target_modules"],
        bias           = "none",
    )
    model     = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    rollout_engine = M4RolloutEngine(base_model, tokenizer, device=device)  # base for rollouts
    scheduler = UCBCurriculumScheduler(cfg["domains"])
    monitor   = SmokeTestMonitor(cfg)

    # ── Data loading ──────────────────────────────────────────────────────────
    domain_data = {
        d: load_domain_samples(d, n=cfg["samples_per_domain"])
        for d in cfg["domains"]
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    wandb.init(project=cfg["wandb_project"], config=cfg, name="m4_smoke_50steps")

    for step in range(cfg["num_steps"]):

        # Sample domain via UCB
        domain = scheduler.sample_domain()
        batch  = _sample_batch(domain_data[domain], cfg["batch_size"])

        prompts       = [item["prompt"]       for item in batch]
        ground_truths = [item["ground_truth"]  for item in batch]
        domains       = [domain] * cfg["batch_size"]

        # Rollout
        rollouts = rollout_engine.generate(
            prompts,
            G              = cfg["G"],
            max_new_tokens = cfg["max_new_tokens"],
            min_new_tokens = cfg["min_new_tokens"],
        )

        # Reward computation
        gdpo_rewards, raw_rewards = score_batch(rollouts, ground_truths, domains)

        # SAN advantages
        advantages = compute_san_advantages(gdpo_rewards, domains)

        # Token-level expansion
        comp_lengths = [[len(rollouts[i]["token_ids"][j]) for j in range(cfg["G"])]
                        for i in range(cfg["batch_size"])]
        token_advs   = expand_advantages_to_tokens(advantages, comp_lengths)

        # Prepare tensors for policy update
        input_ids, attention_mask, completion_mask, old_logprobs = \
            _pack_rollouts(rollouts, tokenizer, device)

        token_adv_tensor = _expand_to_seq(token_advs, input_ids.shape, completion_mask)

        # GRPO loss
        model.train()
        optimizer.zero_grad()
        losses = grpo_loss(
            policy_model     = model,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            completion_mask  = completion_mask,
            advantages       = token_adv_tensor,
            old_logprobs     = old_logprobs,
            beta             = cfg["beta"],
            clip_eps         = cfg["clip_eps"],
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update curriculum
        domain_advantages = {domain: gdpo_rewards.flatten().tolist()}
        scheduler.update(domain_advantages)

        # Monitoring
        alerts = monitor.log_step(step, losses, rollouts, raw_rewards, domain_advantages)
        if alerts:
            print(f"[Step {step}] ALERTS: {alerts}")

        if step % 10 == 0:
            print(f"[Step {step:3d}] loss={losses['loss']:.4f} "
                  f"kl={losses['kl']:.4f} "
                  f"entropy={losses['entropy']:.4f} "
                  f"outcome_r={raw_rewards[0].mean():.3f} "
                  f"domain={domain}")

    print("\n✓ Smoke test completed (50 steps)")
    print("Check W&B dashboard for training curves.")
    wandb.finish()


if __name__ == "__main__":
    run_smoke_test()
```

#### [ ] 7.3 — Run smoke test

```bash
python m4/m4_train.py
```

**Expected duration:** ~90 minutes on M4 (0.5B model, 50 steps, G=4, max_tokens=200)

**Watch for these signals in the terminal output:**

```
HEALTHY state — all green:
  [Step  0] loss=1.xxxx kl=0.000 entropy=3.xxx outcome_r=0.0xx
  [Step 10] loss=0.8xxx kl=0.003 entropy=3.1xx outcome_r=0.1xx  ← KL rising (normal)
  [Step 20] loss=0.7xxx kl=0.008 entropy=3.0xx outcome_r=0.2xx  ← outcome improving
  [Step 50] loss=0.5xxx kl=0.012 entropy=2.9xx outcome_r=0.3xx  ← steady learning

RED FLAGS requiring investigation before Kaggle:
  kl = negative at any step          → KL formula bug (sign error)
  entropy = 0.xxx after step 20      → entropy collapse (increase β)
  outcome_r = 0.000 after step 30    → structural reward blocking all rollouts
  loss = NaN at any step             → gradient explosion (reduce LR)
  ALERTS: [LEARNING_STALLED]         → check domain curriculum, increase G
  All completions identical          → temperature too low, increase to 0.9+
```

---

## PHASE 4 — SMOKE TEST VALIDATION CHECKLIST

Run through these checks after the smoke test completes.

#### [ ] 8.1 — W&B dashboard verification

Open `stratarl_m4_smoke` project in W&B and verify:

```
learning/mean_outcome_reward:     upward trend (even slight) from step 0 to 50
learning/mean_structural_reward:  > 0.1 by step 20 (model learning format)
train/kl:                         positive and bounded (0.001 to 0.05)
train/entropy:                    NOT monotonically decreasing to near-zero
train/prefix_diversity:           > 0.3 (model exploring diverse reasoning paths)
train/reward_zero_std_frac:       < 0.5 (not most groups collapsed)
curriculum/*/ucb_weight:          varies over time (scheduler is active)
```

#### [ ] 8.2 — Qualitative output inspection

```python
# Run this after smoke test to inspect actual model outputs

from m4.m4_rollout_engine import build_m4_engine

engine = build_m4_engine("Qwen/Qwen2.5-0.5B-Instruct")

test_questions = [
    ("What is 15 × 7 + 8?", "gsm8k"),
    ("Would a penguin survive in the Sahara?", "strategyqa"),
]

for question, domain in test_questions:
    rollouts = engine.generate([question], G=2, max_new_tokens=150)
    print(f"\n{'='*60}")
    print(f"Q ({domain}): {question}")
    for i, completion in enumerate(rollouts[0]["completions"]):
        print(f"\n--- Completion {i+1} ---")
        print(completion[:400])

# LOOK FOR:
#   ✓ <think> block present
#   ✓ Domain template tags present (<decompose>, <compute>, etc.)
#   ✓ Reasoning is non-trivial (not just "the answer is X")
#   ✓ <answer> block present with plausible answer
#
# RED FLAGS:
#   ✗ <think></think> (empty think block)
#   ✗ No template tags
#   ✗ Same completion for both rollouts (no diversity)
#   ✗ Garbage/repetitive content
```

#### [ ] 8.3 — Gradient health check

```python
# After smoke test, verify gradients flowed correctly

import torch
from peft import PeftModel

# Load the saved checkpoint (add checkpoint saving to m4_train.py)
# model.save_pretrained("checkpoints/smoke_step50")

# Check: LoRA adapter weights have changed from init
# Load fresh model and compare adapter weights
# If all adapters are at init values, gradients did not flow
```

#### [ ] 8.4 — KL sanity check in live training

```python
# During smoke test, print this every 10 steps to verify KL is positive:

print(f"KL value: {losses['kl'].item():.6f}")
# Must be positive. If negative: KL formula bug is present.
# If always exactly 0.0: old_logprobs are not being used correctly.
```

---

## PHASE 5 — ARCHITECTURE DECISIONS FOR KAGGLE

After smoke test passes, note any parameter adjustments needed before Kaggle:

#### [ ] 9.1 — Record M4 smoke test findings

```
# Fill this in after smoke test:

OUTCOME_REWARD_AT_STEP_50: ___________   (target: > 0.15)
STRUCTURAL_REWARD_AT_STEP_20: _________  (target: > 0.10)
KL_RANGE_OBSERVED: ___________           (target: 0.001 - 0.05)
ENTROPY_FINAL: ___________               (target: > 2.5)
PREFIX_DIVERSITY_MEAN: ___________       (target: > 0.3)
ANY_ALERTS_TRIGGERED: ___________        (ideal: none)
COMPLETION_QUALITY: ___________          (subjective: good/medium/poor)

PARAMETER ADJUSTMENTS FOR KAGGLE:
  β: ___________      (default: 0.01, increase if entropy collapses)
  temperature: _____  (default: 0.85, increase if diversity < 0.3)
  G: ___________      (default: 8 on P100, M4 uses 4)
  max_new_tokens: ___ (default: 2048 on P100, M4 uses 200 for speed)
```

#### [ ] 9.2 — Confirm Kaggle-specific differences are documented

```
DIFFERENCES BETWEEN M4 AND KAGGLE CONFIGS:

M4 (smoke test)             →  Kaggle P100 (production)
─────────────────────────────────────────────────────────
Qwen2.5-0.5B-Instruct       →  Qwen2.5-3B-Instruct
HF generate() + MPS         →  vLLM + CUDA (PagedAttention)
G=4                         →  G=8
max_new_tokens=200           →  max_new_tokens=2048
bfloat16 full precision      →  4-bit QLoRA (Unsloth)
PEFT LoRA (r=8)              →  Unsloth LoRA (r=32)
50 steps                     →  1000 steps
No SFT warmup               →  Phase 0 SFT (Phase 1: ST-GRPO)
Logprobs: HF output.scores   →  vLLM SamplingParams(logprobs=1)
Single domain per step       →  UCB multi-domain curriculum
```

---

## QUICK REFERENCE: COMMON ERRORS AND FIXES

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: Expected all tensors on same device` | model on MPS, tensors on CPU | Add `.to(device)` before forward pass |
| `KL is negative at every step` | Sign-inverted KL formula | Verify `kl = exp(old_logp) × (old_logp - new_logp)` |
| `AssertionError: CRITICAL ALIGNMENT FAILURE` | logprob/token_id length mismatch | Check `output.scores` length vs `sequences[prompt_len:]` |
| `mean_outcome_reward = 0.000 for 30+ steps` | All structural rewards failing | Lower `DOMAIN_MIN_THINK_CHARS`, check template tags in prompts |
| `entropy < 0.5 after 20 steps` | Entropy collapse | Increase β to 0.04 or temperature to 0.95 |
| `All G completions identical` | Temperature too low | Increase to 0.9, check `do_sample=True` |
| `NaN in loss` | Gradient explosion | Reduce LR to 1e-6, verify `clip_grad_norm_` active |
| `prefix_diversity < 0.1` | BLE — beginning lock-in | Increase temperature, check min_new_tokens not too large |
| `pytest: ModuleNotFoundError` | Wrong working directory | `cd stratarl && export PYTHONPATH=.` |
| `MPS backend is not available` | Wrong torch version | `pip install --upgrade torch` |
| `bitsandbytes error on MPS` | Trying to use CUDA quantization | Do NOT use `load_in_4bit=True` on M4 |

---

## DONE CRITERIA

The M4 validation is complete and Kaggle training is cleared when:

- [ ] All component unit tests pass (pytest tests/ — 0 failures)
- [ ] Integration reward test passes (realistic completions scored correctly)
- [ ] Alignment assertion never fires during 50-step smoke test
- [ ] KL is positive at every logged step
- [ ] mean_outcome_reward shows upward trend by step 30
- [ ] Structural reward > 0.10 by step 20
- [ ] Entropy stays above 2.0 throughout
- [ ] Prefix diversity > 0.25 (model exploring diverse reasoning)
- [ ] No NaN/Inf in any tensor at any step
- [ ] Qualitative output inspection: completions have structure + domain tags
- [ ] M4 findings recorded in Section 9.1
- [ ] Kaggle config adjustments finalized

**When all boxes checked:** push to Kaggle, run EXP_01 with Qwen2.5-3B-Instruct.

---

*StrataRL M4 Validation Runbook v1.0*
*Agent-optimized: each section is independently executable.*
*Binary done-checks throughout — no ambiguous completion states.*