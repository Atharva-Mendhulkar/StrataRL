# build.md — StrataRL Implementation Instructions
## Version 3.0 | Complete Fix Manifest + Empirical Validation Layer
## Status: Authoritative implementation source

> **HOW TO USE THIS FILE**
> This is the single source of truth for every code change needed.
> Read the entire document before writing any code.
> Execute phases in strict order — each phase has a hard dependency on the previous.
> Never skip a phase. Never apply a patch without running its test.
> If a test fails, fix it before continuing.

---

## CONTEXT: WHERE THE SYSTEM STANDS

The StrataRL infrastructure has passed numerical and stability audits.
What it has NOT yet done is demonstrate benchmark improvement on:

- GSM8K (target: +5% over baseline 86.7%)
- MMLU (target: +5% over baseline 64.4%)
- StrategyQA (target: +5% over baseline ~65%)

Everything in this document is ordered to close that gap.
Phases 1–3 fix open code issues. Phase 4 adds the empirical layer.
Phase 5 runs the actual experiments. Phase 6 packages results.

---

## REPOSITORY STRUCTURE ASSUMED

```
stratarl/
├── rewards/
│   ├── reward_engine.py
│   └── outcome_verifiers.py
├── training/
│   ├── policy_update.py
│   ├── advantage.py
│   ├── recompute.py           ← CREATE if not exists
│   └── domain_guard.py        ← CREATE if not exists
├── monitoring/
│   └── monitor.py
├── curriculum/
│   └── ucb_scheduler.py
├── rollout/
│   └── rollout_engine.py
├── eval/
│   └── benchmark_eval.py      ← CREATE
├── scripts/
│   ├── audit_config.py
│   ├── generate_report.py
│   └── run_ablation.py        ← CREATE
├── configs/
│   ├── m4_config.yaml
│   └── exp_01_kaggle.yaml     ← CREATE via script
├── m4/
│   └── m4_train.py
├── tests/
│   ├── test_patch_i1.py       ← CREATE
│   ├── test_patch_i2.py       ← CREATE
│   ├── test_patch_i3.py       ← CREATE
│   ├── test_patch_i4.py       ← CREATE
│   ├── test_patch_i5.py       ← CREATE
│   └── test_patch_i7.py       ← CREATE
└── CLAUDE.md                  ← THIS FILE
```

---

## PHASE 1 — INFRASTRUCTURE PATCHES (I-1 through I-9)

Apply in exact order. Run the associated test after each patch.
A failed test means the patch is wrong. Fix the patch, not the test.

---

### PATCH I-1: GDPO Noise — Scale to Clip Range + Annealing

**File:** `rewards/reward_engine.py`

**Problem being fixed:**
Old noise was fixed at `±0.01` regardless of reward scale.
After reward clipping to `[-2.0, 2.0]`, the noise was 0.5% of the reward range —
too weak to break plateaus. Also, constant noise creates temporal drift bias:
the perturbation structure is predictable and the model can learn to exploit it.

**Exact changes:**

1. Add these constants at the top of `reward_engine.py`, replacing any existing GDPO constants:

```python
REWARD_CLIP_RANGE    = 2.0
GDPO_NOISE_FRACTION  = 0.005        # noise = ±(fraction × clip_span)
GDPO_CLIP_SPAN       = REWARD_CLIP_RANGE * 2   # = 4.0
GDPO_ZERO_VAR_THRESH = 1e-2         # RAISED from 1e-3: partial credit has real std
```

2. Replace `gdpo_normalize()` entirely with:

```python
def get_gdpo_noise_magnitude(step: int) -> float:
    anneal_steps = 200
    noise_start  = GDPO_NOISE_FRACTION   # 0.005
    noise_floor  = 0.001
    if step >= anneal_steps:
        fraction = noise_floor
    else:
        fraction = noise_start - (noise_start - noise_floor) * (step / anneal_steps)
    return fraction * GDPO_CLIP_SPAN


def gdpo_normalize(x: torch.Tensor, step: int = 0) -> torch.Tensor:
    """
    For rows with std >= GDPO_ZERO_VAR_THRESH: standard Z-normalization.
    For rows with std < GDPO_ZERO_VAR_THRESH: inject annealed synthetic ranking noise.
    Noise magnitude = GDPO_NOISE_FRACTION × GDPO_CLIP_SPAN, annealed over 200 steps.
    """
    B, G = x.shape
    z    = torch.zeros_like(x)
    eta  = get_gdpo_noise_magnitude(step)

    for i in range(B):
        row_std = x[i].std().item()
        if row_std >= GDPO_ZERO_VAR_THRESH:
            mu   = x[i].mean()
            sig  = x[i].std() + 1e-8
            z[i] = (x[i] - mu) / sig
        else:
            noise = torch.linspace(-eta, eta, steps=G, device=x.device)
            perm  = torch.randperm(G, device=x.device)
            z[i]  = noise[perm]
    return z
```

3. Add `clip_rewards()` function:

```python
def clip_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """Clip BEFORE GDPO and SAN. This is the mandated operation order."""
    return torch.clamp(rewards, -REWARD_CLIP_RANGE, REWARD_CLIP_RANGE)
```

**Test to run:**

```python
# tests/test_patch_i1.py
import torch, pytest
from rewards.reward_engine import gdpo_normalize, get_gdpo_noise_magnitude, GDPO_CLIP_SPAN

def test_noise_scales_with_clip_range():
    noise = get_gdpo_noise_magnitude(0)
    assert abs(noise - 0.005 * GDPO_CLIP_SPAN) < 1e-6

def test_noise_anneals_monotonically():
    magnitudes = [get_gdpo_noise_magnitude(s) for s in range(0, 201, 10)]
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] >= magnitudes[i+1]

def test_noise_floor_stable_after_200():
    assert abs(get_gdpo_noise_magnitude(200) - get_gdpo_noise_magnitude(500)) < 1e-8

def test_partial_credit_uses_znorm_not_noise():
    partial = torch.tensor([[0.4, 0.6, 0.4, 0.6]])
    z = gdpo_normalize(partial, step=0)
    assert z.abs().max().item() > 0.5  # Z-norm produces wide range

def test_zero_variance_uses_noise():
    same = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    z = gdpo_normalize(same, step=0)
    assert z.abs().max().item() < 0.1  # noise is small magnitude
```

```bash
pytest tests/test_patch_i1.py -v
# MUST PASS: 5/5 before continuing
```

---

### PATCH I-2: π_ref = π_old — Remove Reference Model, Fix KL, Add Assertions

**File:** `training/policy_update.py`

**Problems being fixed:**
1. `ref_model` as a separate parameter wastes 1.8GB VRAM
2. KL formula was sign-inverted (`log_ratio - (ratio-1)` is always ≤ 0)
3. No assertion verifying prompt-region logprobs are zeroed
4. `raw_kl_mean` not logged separately from normalized KL

**Add this ADR comment at the top of the file:**

```python
"""
ARCHITECTURAL DECISION RECORD: π_ref = π_old
=============================================
π_ref is defined as rollout-time logprobs (captured via vLLM logprobs=1).
No separate frozen SFT model is loaded or maintained.
This saves 1.8GB VRAM and uses the mathematically correct PPO formulation:
KL penalizes drift from the policy that GENERATED the rollouts.
This decision is irrevocable. Do not add ref_model back.
"""
```

**Replace `grpo_loss()` entirely:**

```python
def grpo_loss(
    policy_model,
    input_ids:        torch.Tensor,
    attention_mask:   torch.Tensor,
    completion_mask:  torch.Tensor,
    advantages:       torch.Tensor,
    old_logprobs:     torch.Tensor,   # rollout-time logprobs = π_ref = π_old
    beta:             float = 0.01,
    clip_eps:         float = 0.2,
    entropy_floor:    float = 0.0,
    entropy_coeff:    float = 0.01,
    recompute_check:  bool  = False,
) -> dict:

    # I-8: Prompt-region assertion — old_logprobs must be zero on prompt tokens
    prompt_mask = (completion_mask == 0).float()
    prompt_logp_contamination = (old_logprobs * prompt_mask).abs().sum().item()
    assert prompt_logp_contamination < 1e-3, (
        f"PROMPT-REGION CONTAMINATION: old_logprobs has non-zero values in prompt "
        f"positions (sum={prompt_logp_contamination:.6f}). "
        f"Check packing logic in utils/tokenize.py."
    )

    # Forward pass
    outputs     = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits      = outputs.logits[:, :-1, :]
    labels      = input_ids[:, 1:]
    log_probs   = F.log_softmax(logits, dim=-1)
    policy_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    old_logp_aligned = old_logprobs[:, 1:].detach()

    # I-8: Shape assertion
    assert old_logp_aligned.shape == policy_logp.shape, (
        f"ALIGNMENT FAILURE: old_logp {old_logp_aligned.shape} != "
        f"policy_logp {policy_logp.shape}. HALTING."
    )

    # Log ratio — clamped in log space to prevent overflow (I from StrataRL audit)
    log_ratio         = torch.clamp(policy_logp - old_logp_aligned, -10.0, 10.0)
    ratio             = torch.exp(log_ratio)

    comp_mask  = completion_mask[:, 1:]
    adv_tokens = advantages[:, 1:]

    # Surrogate loss
    surr1       = ratio * adv_tokens
    surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tokens
    policy_loss = -torch.min(surr1, surr2)

    # KL — CORRECT formula (fixed sign, uses actual old_logprobs)
    # = p_old(t) * log(p_old(t) / p_new(t)) per token
    # Always >= 0 in expectation. Individual tokens may be negative — do not clamp.
    raw_kl_per_token = torch.exp(old_logp_aligned) * (old_logp_aligned - policy_logp)

    # I-9: Track raw KL for absolute drift monitoring (separate from normalized)
    kl_scale  = raw_kl_per_token.abs().mean().detach() + 1e-8
    kl_norm   = raw_kl_per_token / kl_scale

    # Entropy
    entropy = -(torch.exp(policy_logp) * policy_logp * comp_mask).sum() / (comp_mask.sum() + 1e-8)

    # Aggregate
    denom             = comp_mask.sum() + 1e-8
    policy_loss_mean  = (policy_loss * comp_mask).sum() / denom
    kl_norm_mean      = (kl_norm * comp_mask).sum() / denom
    raw_kl_mean       = (raw_kl_per_token * comp_mask).sum() / denom    # I-9
    clip_frac         = ((ratio - 1).abs() > clip_eps).float()
    clip_frac_mean    = (clip_frac * comp_mask).sum() / denom

    total_loss = policy_loss_mean + beta * kl_norm_mean

    entropy_deficit = torch.tensor(0.0, device=input_ids.device)
    if entropy_floor > 0.0:
        entropy_deficit = F.relu(torch.tensor(entropy_floor, device=input_ids.device) - entropy)
        total_loss      = total_loss - entropy_coeff * entropy_deficit

    return {
        "loss":            total_loss,
        "policy_loss":     policy_loss_mean,
        "kl_norm":         kl_norm_mean,
        "raw_kl_mean":     raw_kl_mean,       # I-9: absolute drift signal
        "kl_scale":        kl_scale,
        "entropy":         entropy,
        "entropy_deficit": entropy_deficit,
        "ratio_mean":      (ratio * comp_mask).sum() / denom,
        "ratio_max":       ratio.max(),
        "clip_frac":       clip_frac_mean,
    }
```

**Test to run:**

```python
# tests/test_patch_i2.py
import inspect, pytest
from training.policy_update import grpo_loss

def test_no_ref_model_parameter():
    sig = inspect.signature(grpo_loss)
    assert "ref_model" not in sig.parameters

def test_old_logprobs_required():
    sig = inspect.signature(grpo_loss)
    assert "old_logprobs" in sig.parameters

def test_raw_kl_mean_in_output():
    src = inspect.getsource(grpo_loss)
    assert '"raw_kl_mean"' in src

def test_prompt_region_assertion_fires():
    import torch
    from unittest.mock import MagicMock
    B, G, seq, vocab = 1, 2, 20, 100
    input_ids       = torch.randint(0, vocab, (B*G, seq))
    attn_mask       = torch.ones(B*G, seq)
    comp_mask       = torch.zeros(B*G, seq)
    comp_mask[:, 10:] = 1
    advantages      = torch.zeros(B*G, seq)
    bad_logprobs    = torch.zeros(B*G, seq)
    bad_logprobs[:, :10] = -0.5   # non-zero in prompt region

    mock_model = MagicMock()
    mock_model.return_value.logits = torch.randn(B*G, seq, vocab)

    with pytest.raises(AssertionError, match="PROMPT-REGION CONTAMINATION"):
        grpo_loss(mock_model, input_ids, attn_mask, comp_mask, advantages, bad_logprobs)
```

```bash
pytest tests/test_patch_i2.py -v
# MUST PASS: 4/4
```

---

### PATCH I-3: Recompute — Periodic Not Per-Step

**File:** `training/recompute.py` (CREATE if not exists)

**Problem:** Teacher-forced recomputation at every step adds 40% wall-time.

**Create this file:**

```python
# training/recompute.py

import torch
import torch.nn.functional as F
from typing import Dict

RECOMPUTE_INTERVAL     = 25
DRIFT_WARN_THRESHOLD   = 0.5
DRIFT_ABORT_THRESHOLD  = 2.0


def should_recompute(step: int) -> bool:
    return step % RECOMPUTE_INTERVAL == 0


@torch.no_grad()
def teacher_forced_recompute(
    policy_model,
    input_ids:       torch.Tensor,
    attention_mask:  torch.Tensor,
    completion_mask: torch.Tensor,
    old_logprobs:    torch.Tensor,
    step:            int,
) -> Dict:
    policy_model.eval()

    outputs     = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits      = outputs.logits[:, :-1, :]
    labels      = input_ids[:, 1:]
    log_probs   = F.log_softmax(logits, dim=-1)
    policy_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    old_aligned = old_logprobs[:, 1:]
    comp_mask   = completion_mask[:, 1:]

    diff        = (policy_logp - old_aligned).abs()
    drift_mean  = (diff * comp_mask).sum() / (comp_mask.sum() + 1e-8)
    drift_max   = diff.max()

    drift_mean_val = drift_mean.item()
    status = "ABORT" if drift_mean_val > DRIFT_ABORT_THRESHOLD else \
             "WARN"  if drift_mean_val > DRIFT_WARN_THRESHOLD  else "OK"

    policy_model.train()

    return {
        "recompute_drift_mean": drift_mean_val,
        "recompute_drift_max":  drift_max.item(),
        "recompute_status":     status,
        "recompute_step":       step,
    }
```

**Add this block to the training loop in `m4/m4_train.py` after `optimizer.step()`:**

```python
from training.recompute import should_recompute, teacher_forced_recompute, DRIFT_ABORT_THRESHOLD

if should_recompute(step):
    rc = teacher_forced_recompute(model, input_ids, attention_mask,
                                   completion_mask, old_logprobs_t, step)
    wandb.log(rc, step=step)
    if rc["recompute_status"] == "ABORT":
        raise RuntimeError(
            f"[Step {step}] RECOMPUTE ABORT: drift={rc['recompute_drift_mean']:.4f}. "
            f"Reduce vllm_sync_interval or check tokenization packing."
        )
```

**Test to run:**

```python
# tests/test_patch_i3.py
from training.recompute import should_recompute, RECOMPUTE_INTERVAL

def test_fires_at_correct_intervals():
    fired = [s for s in range(200) if should_recompute(s)]
    assert fired == list(range(0, 200, RECOMPUTE_INTERVAL))

def test_overhead_under_5pct():
    fired = sum(1 for s in range(100) if should_recompute(s))
    assert fired / 100 <= 0.05

def test_interval_in_reasonable_range():
    assert 10 <= RECOMPUTE_INTERVAL <= 50
```

```bash
pytest tests/test_patch_i3.py -v
# MUST PASS: 3/3
```

---

### PATCH I-4: Δ_O/S Tracker — Diverse Nonsense Detection

**File:** `monitoring/monitor.py`

**Problem:** Model can achieve high structural reward with near-zero outcome reward
by learning formatting without reasoning. No automated detection existed.

**Add `DeltaOSTracker` class:**

```python
# Add to monitoring/monitor.py

from collections import deque
from typing import Optional
import numpy as np

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
```

**In `monitor.log_step()` — add these lines:**

```python
# In log_step(), after extracting mean_outcome and mean_structural:
delta_os_metrics = self.delta_os_tracker.update(mean_outcome, mean_structural, step)
metrics.update(delta_os_metrics)

# In alert section:
if delta_os_metrics["delta_os/status"] == "ATTACK":
    alerts.append("DIVERSE_NONSENSE_ATTACK")
    # Apply weight override in reward engine on next step
```

**Test to run:**

```python
# tests/test_patch_i4.py
from monitoring.monitor import DeltaOSTracker, DIVERSE_NONSENSE_TRIGGER, DELTA_OS_ALERT_THRESHOLD

def test_no_alert_healthy_training():
    t = DeltaOSTracker()
    for s in range(60):
        r = t.update(0.7, 0.6, s)
    assert r["delta_os/status"] == "OK"

def test_attack_detected_after_trigger():
    t = DeltaOSTracker()
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        r = t.update(0.001, 0.8, s)
    assert r["delta_os/status"] == "ATTACK"

def test_no_false_alarm_before_trigger():
    t = DeltaOSTracker()
    results = [t.update(0.001, 0.8, s) for s in range(DIVERSE_NONSENSE_TRIGGER - 1)]
    assert results[-1]["delta_os/status"] != "ATTACK"

def test_recovery_clears_attack():
    t = DeltaOSTracker()
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        t.update(0.001, 0.8, s)
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5, DIVERSE_NONSENSE_TRIGGER + 60):
        r = t.update(0.7, 0.8, s)
    assert r["delta_os/status"] != "ATTACK"

def test_weight_override_active_during_attack():
    t = DeltaOSTracker()
    assert t.get_outcome_weight_override() is None
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        t.update(0.001, 0.8, s)
    assert t.get_outcome_weight_override() == 0.85
```

```bash
pytest tests/test_patch_i4.py -v
# MUST PASS: 5/5
```

---

### PATCH I-5: SAN Threshold + Partial-Credit Dampening

**File:** `training/advantage.py`

**Problem:** SAN threshold `1e-3` misfired on partial-credit rewards with real signal.
Also, low-variance partial credit was being amplified to unit variance — treating noise
as strong signal.

**Replace `compute_san_advantages()` / `compute_stratified_advantages()` with:**

```python
# training/advantage.py

import torch
from typing import List

LENGTH_NORM_CLAMP   = 512
ADVANTAGE_CLIP      = 5.0
SAN_ZERO_VAR_THRESH = 1e-2    # raised from 1e-3
SAN_LOW_VAR_THRESH  = 0.05    # new: dampening threshold for weak partial credit


def compute_san_advantages(
    rewards: torch.Tensor,   # [B, G]
    domains: List[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Three normalization regimes:
    std < 1e-2:        center-without-scale (true zero variance)
    1e-2 <= std < 0.05: dampen to prevent ghost amplification of partial credit
    std >= 0.05:       full Z-normalization
    """
    advantages    = torch.zeros_like(rewards)
    unique_domains = set(domains)

    for domain in unique_domains:
        idx = [i for i, d in enumerate(domains) if d == domain]
        if len(idx) < 2:
            advantages[idx] = 0.0
            continue

        stratum = rewards[idx]
        flat    = stratum.flatten()
        mu      = flat.mean()
        sigma   = flat.std()

        if sigma < SAN_ZERO_VAR_THRESH:
            centered         = stratum - mu
            advantages[idx]  = torch.clamp(centered, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        elif sigma < SAN_LOW_VAR_THRESH:
            damping          = sigma.item() / SAN_LOW_VAR_THRESH
            normalized       = (stratum - mu) / (sigma + eps)
            advantages[idx]  = torch.clamp(normalized * damping, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        else:
            normalized       = (stratum - mu) / (sigma + eps)
            advantages[idx]  = torch.clamp(normalized, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    return advantages


def expand_advantages_to_tokens(
    advantages: torch.Tensor,
    completion_lengths: List[List[int]],
    use_length_norm: bool = True,
) -> torch.Tensor:
    token_advantages = []
    for b in range(advantages.shape[0]):
        for g in range(advantages.shape[1]):
            adv    = advantages[b, g].item()
            length = completion_lengths[b][g]
            if use_length_norm and length > 0:
                clamped = min(length, LENGTH_NORM_CLAMP)
                adv     = adv / (clamped ** 0.5)
            token_advantages.extend([adv] * length)
    return torch.tensor(token_advantages, dtype=torch.float32)
```

**Test to run:**

```python
# tests/test_patch_i5.py
import torch, pytest
from training.advantage import compute_san_advantages, SAN_ZERO_VAR_THRESH
from rewards.reward_engine import GDPO_ZERO_VAR_THRESH

def test_zero_variance_centers_without_scaling():
    r = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.abs().max().item() < 1e-5

def test_partial_credit_is_dampened():
    r = torch.tensor([[0.49, 0.51, 0.49, 0.51]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.std().item() < 0.8

def test_high_variance_full_normalization():
    r = torch.tensor([[0.0, 1.0, 0.0, 1.0]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.std().item() > 0.5

def test_threshold_matches_gdpo():
    assert SAN_ZERO_VAR_THRESH == GDPO_ZERO_VAR_THRESH, (
        f"Mismatch: SAN={SAN_ZERO_VAR_THRESH}, GDPO={GDPO_ZERO_VAR_THRESH}"
    )

def test_cross_stratum_bias_prevented():
    r = torch.tensor([
        [0.9, 0.9, 0.8, 0.9], [0.9, 0.9, 0.8, 0.9],
        [0.4, 0.3, 0.4, 0.5], [0.4, 0.3, 0.4, 0.5],
    ])
    d = ["gsm8k", "gsm8k", "strategyqa", "strategyqa"]
    a = compute_san_advantages(r, d)
    assert abs(a[2:4].mean().item()) < 0.3
```

```bash
pytest tests/test_patch_i5.py -v
# MUST PASS: 5/5
```

---

### PATCH I-7: Batch Domain Homogeneity Assertion

**File:** `training/domain_guard.py` (CREATE)

```python
# training/domain_guard.py

def assert_batch_domain_homogeneity(domains: list):
    """
    All prompts in a batch must share the same domain.
    SAN requires >= 2 prompts per domain to compute Z-statistics.
    Mixed-domain batches produce zero advantages for singleton domains,
    wasting the entire training step.
    """
    unique = set(domains)
    assert len(unique) == 1, (
        f"DOMAIN HETEROGENEITY: batch contains {len(unique)} domains: {unique}. "
        f"UCB scheduler must sample one domain per step, then fill the batch from it."
    )
```

**Add this call to the training loop immediately after sampling the batch:**

```python
from training.domain_guard import assert_batch_domain_homogeneity
assert_batch_domain_homogeneity(domains)
```

**Test to run:**

```python
# tests/test_patch_i7.py
import pytest
from training.domain_guard import assert_batch_domain_homogeneity

def test_homogeneous_passes():
    assert_batch_domain_homogeneity(["gsm8k"] * 4)

def test_mixed_raises():
    with pytest.raises(AssertionError, match="DOMAIN HETEROGENEITY"):
        assert_batch_domain_homogeneity(["gsm8k", "mmlu", "gsm8k"])
```

```bash
pytest tests/test_patch_i7.py -v
# MUST PASS: 2/2
```

---

### PATCHES I-6, I-8, I-9: Confirmation Checklist

These were implemented inline in the patches above. Verify each:

- **I-6** (GDPO temporal drift annealing): Confirmed present in `get_gdpo_noise_magnitude()` — annealing over 200 steps to `noise_floor=0.001`.
- **I-8** (Prompt-region logprob assertion): Confirmed present in `grpo_loss()` — `assert prompt_logp_contamination < 1e-3`.
- **I-9** (raw_kl_mean logged separately): Confirmed present in `grpo_loss()` return dict and `monitor.log_step()`.

---

## PHASE 2 — RUN FULL TEST SUITE

```bash
pytest tests/ -v --tb=short 2>&1 | tee reports/test_suite_results.txt
```

**Acceptance criteria — DO NOT proceed if any fail:**
- Zero failures
- Zero errors
- Minimum 22 tests collected and passing
- No `XFAIL` that should be passing

```bash
# Coverage check
pytest tests/ --cov=rewards --cov=training --cov=monitoring \
  --cov-report=term-missing 2>&1 | tee reports/coverage_report.txt

# Minimum:
#   rewards/reward_engine.py  > 80%
#   training/advantage.py     > 85%
#   training/policy_update.py > 75%
#   monitoring/monitor.py     > 70%
```

---

## PHASE 3 — EMPIRICAL EVALUATION LAYER

**This is the most critical phase. The infrastructure is hardened. Now prove it improves reasoning.**

### 3.1 Create Evaluation Pipeline

**File:** `eval/benchmark_eval.py` (CREATE)

```python
# eval/benchmark_eval.py

import torch
import numpy as np
import re
from datasets import load_dataset
from typing import Dict, List, Tuple
from sympy import sympify, simplify
from collections import Counter

# ── Answer extractors ─────────────────────────────────────────────────────────

THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE  = re.compile(r"\\boxed\{([^}]+)\}")

def extract_answer(completion: str) -> str | None:
    m = ANSWER_RE.search(completion)
    if m: return m.group(1).strip()
    m = BOXED_RE.search(completion)
    if m: return m.group(1).strip()
    return None

def extract_think_length(completion: str) -> int:
    m = THINK_RE.search(completion)
    if not m: return 0
    return len(m.group(1).split())  # word count

# ── Verifiers ─────────────────────────────────────────────────────────────────

def verify_math(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    try:
        pred_expr  = sympify(predicted, evaluate=True)
        truth_expr = sympify(ground_truth, evaluate=True)
        if simplify(pred_expr - truth_expr) == 0: return True
    except Exception: pass
    try:
        if abs(float(predicted) - float(ground_truth)) < 1e-6: return True
    except (ValueError, TypeError): pass
    return predicted.strip().lower() == ground_truth.strip().lower()

def verify_mcq(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    return predicted.strip().upper()[:1] == ground_truth.strip().upper()[:1]

def verify_bool(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    p = predicted.strip().lower()
    g = ground_truth.strip().lower()
    # Normalize yes/no/true/false
    p_bool = p in ("yes", "true", "1")
    g_bool = g in ("yes", "true", "1")
    return p_bool == g_bool


# ── Benchmark configs ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "gsm8k": {
        "hf_path":    "gsm8k",
        "hf_name":    "main",
        "split":      "test",
        "n_samples":  500,     # subset for speed; full is 1319
        "verifier":   verify_math,
        "answer_key": "answer",
        "is_primary": True,
        "baseline":   0.867,   # Qwen2.5-3B baseline
        "target":     0.917,   # +5%
    },
    "mmlu": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "all",
        "split":      "test",
        "n_samples":  500,
        "verifier":   verify_mcq,
        "answer_key": "answer",   # integer 0-3; map to A-D
        "is_primary": True,
        "baseline":   0.644,
        "target":     0.694,
    },
    "strategyqa": {
        "hf_path":    "wics/strategy-qa",
        "hf_name":    None,
        "split":      "test",
        "n_samples":  490,
        "verifier":   verify_bool,
        "answer_key": "answer",
        "is_primary": True,
        "baseline":   0.650,
        "target":     0.700,
    },
}

MMLU_INT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

def format_gsm8k_prompt(item: dict) -> Tuple[str, str]:
    q  = item["question"]
    gt = item["answer"].split("####")[-1].strip()
    prompt = (
        "You are a precise math reasoning assistant.\n"
        "Solve step by step. Show all work inside <think> tags. "
        "Place only the final numeric answer inside <answer> tags.\n\n"
        f"Question: {q}\n"
    )
    return prompt, gt

def format_mmlu_prompt(item: dict) -> Tuple[str, str]:
    q       = item["question"]
    choices = item["choices"]
    gt_int  = item["answer"]
    gt      = MMLU_INT_TO_LETTER[gt_int]
    options = "\n".join(f"{MMLU_INT_TO_LETTER[i]}. {c}" for i, c in enumerate(choices))
    prompt  = (
        "You are a knowledgeable assistant.\n"
        "Answer the following multiple choice question. "
        "Reason briefly inside <think> tags. "
        "Write only the letter (A/B/C/D) inside <answer> tags.\n\n"
        f"Question: {q}\n{options}\n"
    )
    return prompt, gt

def format_strategyqa_prompt(item: dict) -> Tuple[str, str]:
    q  = item["question"]
    gt = "yes" if item["answer"] else "no"
    prompt = (
        "You are a logical reasoning assistant.\n"
        "Answer the following yes/no question. "
        "Reason briefly inside <think> tags. "
        "Write only 'yes' or 'no' inside <answer> tags.\n\n"
        f"Question: {q}\n"
    )
    return prompt, gt

FORMATTERS = {
    "gsm8k":       format_gsm8k_prompt,
    "mmlu":        format_mmlu_prompt,
    "strategyqa":  format_strategyqa_prompt,
}


# ── Evaluation runner ─────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    def __init__(self, generate_fn):
        """
        generate_fn: callable(prompts: List[str], temperature: float, max_tokens: int)
                     -> List[str] (completions)
        Use temperature=0.0 for greedy (primary metric).
        Use temperature=0.7, n=5 for best-of-5 (secondary metric, math only).
        """
        self.generate = generate_fn

    def run_benchmark(
        self,
        bench_name: str,
        greedy_only: bool = False,
    ) -> Dict:
        cfg       = BENCHMARKS[bench_name]
        formatter = FORMATTERS[bench_name]

        ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
        ds = ds.select(range(min(cfg["n_samples"], len(ds))))

        greedy_correct = 0
        best5_correct  = 0
        think_lengths  = []
        total          = len(ds)

        for item in ds:
            prompt, gt = formatter(item)

            # Greedy eval
            greedy_completions = self.generate([prompt], temperature=0.0, max_tokens=512)
            pred = extract_answer(greedy_completions[0])
            if cfg["verifier"](pred, gt):
                greedy_correct += 1
            think_lengths.append(extract_think_length(greedy_completions[0]))

            # Best-of-5 (math benchmarks only, skip if greedy_only)
            if not greedy_only and bench_name in ("gsm8k",):
                sampled = self.generate([prompt], temperature=0.7, max_tokens=512, n=5)
                if any(cfg["verifier"](extract_answer(c), gt) for c in sampled):
                    best5_correct += 1

        greedy_acc  = greedy_correct / total
        best5_acc   = best5_correct  / total if not greedy_only else None
        delta       = greedy_acc - cfg["baseline"]
        target_met  = greedy_acc >= cfg["target"]
        avg_think   = np.mean(think_lengths)

        return {
            "benchmark":    bench_name,
            "n_samples":    total,
            "greedy_acc":   greedy_acc,
            "best5_acc":    best5_acc,
            "baseline":     cfg["baseline"],
            "target":       cfg["target"],
            "delta":        delta,
            "target_met":   target_met,
            "avg_think_len": avg_think,
        }

    def run_all(self, step: int, greedy_only: bool = False) -> Dict:
        results = {}
        for bench in BENCHMARKS:
            r = self.run_benchmark(bench, greedy_only=greedy_only)
            results[bench] = r
            print(
                f"[Step {step}] {bench}: {r['greedy_acc']:.3f} "
                f"(baseline: {r['baseline']:.3f}, delta: {r['delta']:+.3f}, "
                f"target_met: {r['target_met']}, avg_think: {r['avg_think_len']:.0f} words)"
            )
        return results
```

### 3.2 Integrate Evaluator into Training Loop

**In `m4/m4_train.py`, add evaluator call:**

```python
from eval.benchmark_eval import BenchmarkEvaluator

# Initialize once (before training loop)
evaluator = BenchmarkEvaluator(generate_fn=rollout_gen.generate_for_eval)

# Inside training loop, every eval_interval steps:
if step % config.eval_interval == 0 and step > 0:
    eval_results = evaluator.run_all(step=step)

    for bench, r in eval_results.items():
        wandb.log({
            f"eval/{bench}/greedy_acc":    r["greedy_acc"],
            f"eval/{bench}/delta":         r["delta"],
            f"eval/{bench}/target_met":    int(r["target_met"]),
            f"eval/{bench}/avg_think_len": r["avg_think_len"],
        }, step=step)

    # MMLU negative control — alert if drops below baseline
    mmlu_acc = eval_results["mmlu"]["greedy_acc"]
    if mmlu_acc < BENCHMARKS["mmlu"]["baseline"] - 0.03:
        wandb.alert(
            title=f"CATASTROPHIC FORGETTING: MMLU={mmlu_acc:.3f}",
            text=f"MMLU dropped {(BENCHMARKS['mmlu']['baseline'] - mmlu_acc)*100:.1f}% "
                 f"below baseline. Increase β or reduce training intensity.",
            level=wandb.AlertLevel.ERROR,
        )
```

**Add `generate_for_eval` method to `RolloutGenerator`:**

```python
# rollout/rollout_engine.py — ADD method

def generate_for_eval(
    self,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 512,
    n: int = 1,
) -> List[str]:
    """
    Simplified generation for evaluation. Returns flat list of completions.
    temperature=0 for greedy. n>1 for best-of-n sampling.
    """
    from vllm import SamplingParams
    params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = self.llm.generate(prompts, params)
    completions = []
    for output in outputs:
        for o in output.outputs:
            completions.append(o.text)
    return completions
```

### 3.3 Baseline Measurement Protocol

**Run this BEFORE any RL training to establish the exact baseline:**

```bash
python scripts/measure_baseline.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --benchmarks gsm8k mmlu strategyqa \
  --output reports/baseline_measurements.json
```

**Create `scripts/measure_baseline.py`:**

```python
# scripts/measure_baseline.py

import argparse, json
from pathlib import Path
from eval.benchmark_eval import BenchmarkEvaluator
from rollout.rollout_engine import RolloutGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--benchmarks",  nargs="+", default=["gsm8k", "mmlu", "strategyqa"])
    parser.add_argument("--output",      default="reports/baseline_measurements.json")
    args = parser.parse_args()

    gen       = RolloutGenerator(args.model, gpu_memory_utilization=0.60)
    evaluator = BenchmarkEvaluator(generate_fn=gen.generate_for_eval)

    results = {}
    for bench in args.benchmarks:
        r = evaluator.run_benchmark(bench, greedy_only=True)
        results[bench] = r
        print(f"{bench}: {r['greedy_acc']:.4f} (expected baseline: {r['baseline']:.4f})")

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nBaseline measurements saved to {args.output}")

if __name__ == "__main__":
    main()
```

**This baseline run must complete before any RL training. Record the exact numbers.**

---

## PHASE 4 — ABLATION EXECUTION PLAN

**Purpose:** Generate the accuracy ablations that turn this into publishable research.
These are not stability ablations — they are accuracy ablations.

### 4.1 Create Ablation Runner

**File:** `scripts/run_ablation.py` (CREATE)

```python
# scripts/run_ablation.py
"""
Runs 6 targeted ablation experiments. Each experiment trains for 500 steps,
evaluates on all 3 benchmarks, and logs the accuracy delta vs baseline.
"""

import argparse, yaml, subprocess, json
from pathlib import Path

ABLATIONS = {
    "EXP_01_CONTROL": {
        "description": "Full system — all mechanisms active",
        "overrides": {},
        "hypothesis": "Highest benchmark accuracy. Reference for all deltas.",
    },
    "EXP_02_NO_SAN": {
        "description": "SAN disabled — global normalization only",
        "overrides": {"disable_san": True},
        "hypothesis": "MMLU/StrategyQA accuracy drops. Domain starvation re-emerges.",
    },
    "EXP_03_G4": {
        "description": "G=4 group size (vs G=8 control)",
        "overrides": {"G": 4},
        "hypothesis": "Accuracy drops due to high-variance advantage estimates.",
    },
    "EXP_04_NO_KL": {
        "description": "β=0 unconstrained exploration",
        "overrides": {"beta_init": 0.0, "beta_fixed": True,
                      "clip_eps": 0.1, "entropy_floor": 0.5},
        "hypothesis": "Higher peak accuracy but risk of entropy collapse.",
    },
    "EXP_05_NO_CURRICULUM": {
        "description": "Uniform domain sampling (UCB disabled)",
        "overrides": {"disable_curriculum": True},
        "hypothesis": "~2% accuracy drop. Weak domains starve gradient.",
    },
    "EXP_06_PHI3": {
        "description": "Phi-3-mini cross-architecture validation",
        "overrides": {"model": "microsoft/Phi-3-mini-4k-instruct"},
        "hypothesis": "Validates that improvements are not Qwen-specific.",
    },
}

def run_ablation(name: str, config: dict, base_config_path: str, steps: int = 500):
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.update(config["overrides"])
    cfg["num_steps"]  = steps
    cfg["run_name"]   = name
    cfg["eval_interval"] = 100

    tmp_path = f"/tmp/ablation_{name}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f)

    result = subprocess.run(
        ["python", "training/train.py", "--config", tmp_path,
         "--run_name", name, "--wandb_project", "stratarl_ablations"],
        capture_output=True, text=True, timeout=28800
    )
    return result.returncode, result.stdout, result.stderr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/exp_01_kaggle.yaml")
    parser.add_argument("--steps",       type=int, default=500)
    parser.add_argument("--exp",         default="all",
                        help="Run specific experiment by name, or 'all'")
    args = parser.parse_args()

    to_run = ABLATIONS if args.exp == "all" else {args.exp: ABLATIONS[args.exp]}

    for name, config in to_run.items():
        print(f"\nStarting: {name} — {config['description']}")
        print(f"  Hypothesis: {config['hypothesis']}")
        rc, stdout, stderr = run_ablation(name, config, args.base_config, args.steps)
        print(f"  Exit code: {rc}")
        if rc != 0:
            print(f"  STDERR: {stderr[-500:]}")
```

### 4.2 Accuracy Ablation Results Table Template

**After running ablations, fill this table in `reports/accuracy_ablations.md`:**

```markdown
# Accuracy Ablation Results

| Experiment | GSM8K | MMLU | StrategyQA | GSM8K Δ | MMLU Δ | SQA Δ | Stability |
|------------|-------|------|------------|---------|--------|-------|-----------|
| Baseline (no RL) | 86.7% | 64.4% | 65.0% | — | — | — | — |
| EXP_01 (Full) | ?% | ?% | ?% | ? | ? | ? | ? |
| EXP_02 (No SAN) | ?% | ?% | ?% | ? | ? | ? | ? |
| EXP_03 (G=4) | ?% | ?% | ?% | ? | ? | ? | ? |
| EXP_04 (No KL) | ?% | ?% | ?% | ? | ? | ? | ? |
| EXP_05 (No Curriculum) | ?% | ?% | ?% | ? | ? | ? | ? |
| EXP_06 (Phi-3-mini) | ?% | ?% | ?% | ? | ? | ? | ? |

## Key Claims (fill after experiments)
- SAN contributes approximately __% to accuracy (EXP_01 vs EXP_02)
- UCB curriculum contributes approximately __% to accuracy (EXP_01 vs EXP_05)
- G=8 vs G=4 accuracy delta: __%
```

---

## PHASE 5 — LATENCY AND EFFICIENCY MEASUREMENTS

**The challenge explicitly requires efficient reasoning. Measure this.**

### 5.1 Add Latency Profiler

**File:** `eval/latency_profiler.py` (CREATE)

```python
# eval/latency_profiler.py

import time
import torch
import numpy as np
from typing import Dict, List

def profile_inference_latency(
    generate_fn,
    prompts: List[str],
    temperatures: List[float] = [0.0, 0.85],
    max_tokens_list: List[int] = [512, 2048],
    n_warmup: int = 3,
    n_measure: int = 20,
) -> Dict:
    """
    Measures:
    - Tokens per second at different generation settings
    - TTFT (time to first token) — not directly measurable with vLLM batch API
    - End-to-end latency per prompt
    - VRAM usage during inference
    """
    results = {}

    for temp in temperatures:
        for max_tok in max_tokens_list:
            key = f"temp{temp}_maxtok{max_tok}"

            # Warmup
            for _ in range(n_warmup):
                generate_fn(prompts[:1], temperature=temp, max_tokens=max_tok)

            # Measure
            latencies = []
            token_counts = []

            for _ in range(n_measure):
                prompt = prompts[_ % len(prompts)]
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                start = time.perf_counter()
                completions = generate_fn([prompt], temperature=temp, max_tokens=max_tok)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end   = time.perf_counter()

                latency = end - start
                n_tokens = len(completions[0].split())  # word count proxy
                latencies.append(latency)
                token_counts.append(n_tokens)

            results[key] = {
                "mean_latency_sec":   np.mean(latencies),
                "p95_latency_sec":    np.percentile(latencies, 95),
                "mean_tokens":        np.mean(token_counts),
                "tokens_per_sec":     np.mean(token_counts) / np.mean(latencies),
                "temperature":        temp,
                "max_tokens":         max_tok,
            }

    # VRAM measurement
    if torch.cuda.is_available():
        results["vram_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        results["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)

    return results


def compare_baseline_vs_rl(
    baseline_gen_fn,
    rl_gen_fn,
    prompts: List[str],
) -> Dict:
    """Compare latency and token efficiency: baseline SFT vs RL-trained model."""
    baseline_profile = profile_inference_latency(baseline_gen_fn, prompts)
    rl_profile       = profile_inference_latency(rl_gen_fn, prompts)

    comparison = {
        "baseline": baseline_profile,
        "rl":       rl_profile,
    }

    # Key ratio: does RL add reasoning tokens that slow things down?
    for key in baseline_profile:
        if isinstance(baseline_profile[key], dict):
            b_tps = baseline_profile[key].get("tokens_per_sec", 1)
            r_tps = rl_profile[key].get("tokens_per_sec", 1)
            comparison[f"overhead_ratio_{key}"] = r_tps / b_tps

    return comparison
```

---

## PHASE 6 — CONFIG GENERATION AND KAGGLE LAUNCH

### 6.1 Run Config Audit

```bash
python scripts/audit_config.py --config configs/exp_01_kaggle.yaml
# Expected: ✓ Config audit passed
```

### 6.2 Pre-Flight Checklist

Before launching on Kaggle P100, verify every item:

**PATCHES**
- [ ] I-1: GDPO noise = `0.005 × GDPO_CLIP_SPAN = ±0.02` at step 0; anneals to `±0.004` by step 200
- [ ] I-2: `policy_update.py` has ADR comment; no `ref_model` argument in `grpo_loss()`
- [ ] I-2: Prompt-region logprob assertion fires on corrupted input (tested)
- [ ] I-3: `RECOMPUTE_INTERVAL = 25`; recompute block in training loop present
- [ ] I-4: `DeltaOSTracker` instantiated in monitor; `delta_os/status` logged to W&B
- [ ] I-4: `get_outcome_weight_override()` connected to reward engine's `w_outcome`
- [ ] I-5: `SAN_ZERO_VAR_THRESH = 1e-2`; confirmed matches `GDPO_ZERO_VAR_THRESH`
- [ ] I-5: Dampening regime present for `1e-2 <= std < 0.05`
- [ ] I-7: `assert_batch_domain_homogeneity()` called in training loop
- [ ] I-8: Prompt-region assertion in `grpo_loss()` (inside I-2)
- [ ] I-9: `raw_kl_mean` in `grpo_loss()` return dict AND logged in monitor

**TESTS**
- [ ] `pytest tests/ -v` — 0 failures, 0 errors, ≥ 22 tests
- [ ] Coverage: all critical modules > 75%

**EMPIRICAL LAYER**
- [ ] Baseline measurements recorded in `reports/baseline_measurements.json`
- [ ] `BenchmarkEvaluator` integrated into training loop; eval every 100 steps
- [ ] Latency profiler attached to post-training evaluation

**KAGGLE PARAMETERS**
- [ ] `beta_phase1 = 0.015` (tighter during first 100 steps on 3B)
- [ ] `beta_switch_step = 100` (not before)
- [ ] `clip_eps_phase1 = 0.15` (not 0.2 for first 100 steps)
- [ ] `G = 8`
- [ ] `load_ref_model = False`
- [ ] `W&B API key` set as Kaggle secret `WANDB_API_KEY`

**MONITORING PLAN (first 100 Kaggle steps)**
- [ ] W&B dashboard open during run
- [ ] Step 10: `raw_kl_mean < 0.05` (if higher, increase `beta_phase1` to `0.02`)
- [ ] Step 25: recompute drift status = `OK` or `WARN`
- [ ] Step 50: `mean_outcome_reward` trending up (not flat)
- [ ] Step 100: GSM8K eval > 86.7% baseline — if not, abort full run, diagnose

**ABORT TRIGGERS**
- [ ] NaN in any loss metric → hard stop
- [ ] `raw_kl_mean > 0.10` for 10 consecutive steps → checkpoint and stop
- [ ] `entropy < 0.02` → hard stop
- [ ] `prefix_diversity < 0.01` → hard stop
- [ ] `DIVERSE_NONSENSE_ATTACK` sustained 100+ steps without `Δ_O/S` recovery

---

## PHASE 7 — RESULT DOCUMENTATION TEMPLATE

After running experiments, fill this section. This is what gets submitted.

### 7.1 Required Claims

Document every number precisely. No vague language like "improved significantly."

```
MODEL: [exact HF model ID used]
BASE CHECKPOINT: [exact revision/commit]
TRAINING STEPS: [exact number]
HARDWARE: [GPU type, count, VRAM]

BENCHMARK RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benchmark   Baseline   Post-RL   Delta   Target   Met?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GSM8K       86.7%      __._%     +_._% ≥91.7%   YES/NO
MMLU        64.4%      __._%     +_._% ≥69.4%   YES/NO
StrategyQA  65.0%      __._%     +_._% ≥70.0%   YES/NO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVALUATION SETTINGS
- Decoding: greedy (temperature=0)
- Prompt format: [exact template used]
- Sample count: [n per benchmark]
- Statistical comparison: [bootstrap CI or just point estimate]

MECHANISM CONTRIBUTIONS (from ablations)
- SAN effect on MMLU:           __% (EXP_01 vs EXP_02)
- Curriculum effect on SQA:     __% (EXP_01 vs EXP_05)
- G=8 vs G=4 on GSM8K:          __% (EXP_01 vs EXP_03)

EFFICIENCY
- Inference latency (greedy):   __ ms/prompt
- Tokens per second:            __ tok/s
- VRAM at inference:            __ GB
- Avg chain-of-thought length:  __ words
```

---

## KNOWN LIMITATIONS — DO NOT HIDE THESE

Include these in any submission or report:

1. **Single seed**: All results are single-seed. Multi-seed runs require 3× compute budget.
2. **Single-token KL approximation**: The KL estimate uses only the chosen token's logprob, not the full vocabulary distribution. This is an inherent limitation of not storing full logit distributions — shared by all practical GRPO implementations.
3. **Evaluation contamination risk**: GSM8K is an old benchmark. There is a non-zero probability that the base model has seen test questions during pre-training. Use StrategyQA and MMLU as the primary contamination-resistant signals.
4. **Smoke test scope**: All M4 smoke tests were run for 50 steps on a small model. The first 100 Kaggle steps on 3B should be treated as calibration, not assumed to replicate M4 behavior.
5. **Structural reward is still partially gameable**: The graded reward gives partial credit for tag presence. A model could theoretically maximize structural reward while still producing low-quality reasoning. `Δ_O/S` monitoring exists to catch this but cannot catch all variants.

---

## COMMAND REFERENCE

```bash
# Run all infrastructure tests
pytest tests/ -v --tb=short

# Measure baseline (run FIRST, before any training)
python scripts/measure_baseline.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --benchmarks gsm8k mmlu strategyqa \
  --output reports/baseline_measurements.json

# Config audit
python scripts/audit_config.py --config configs/exp_01_kaggle.yaml

# M4 smoke test (50 steps, local)
python m4/m4_train.py --config m4/m4_config.yaml \
  --wandb_project stratarl_m4_v2 \
  --run_name smoke_v2_allpatches

# Run single ablation
python scripts/run_ablation.py --exp EXP_01_CONTROL --steps 500

# Run all ablations
python scripts/run_ablation.py --all --steps 500

# Generate Kaggle config
python scripts/generate_kaggle_config.py

# Kaggle launch (run inside Kaggle notebook)
python training/train.py \
  --config configs/exp_01_kaggle.yaml \
  --run_name EXP_01_qwen3b_1000steps \
  --wandb_project stratarl_kaggle_3b
```

---

## WHAT SUCCESS LOOKS LIKE

The project is complete when:

1. All 22+ tests pass with zero failures
2. Baseline measurements are recorded for all 3 benchmarks
3. Post-RL measurements show ≥+5% on at least 2 of 3 benchmarks
4. Accuracy ablations table is filled in with actual numbers
5. `raw_kl_mean` stayed below 0.10 throughout the full run
6. `Δ_O/S` never declared a sustained attack in the production run
7. MMLU did not drop more than 3% from baseline (no catastrophic forgetting)

If benchmarks do not improve after 500 steps, the most likely causes in priority order:
1. Baseline measurement was wrong (re-run with identical prompt format)
2. Reward signal too sparse (check `mean_outcome_reward` — if < 0.05 after 100 steps, the model is solving almost nothing)
3. Structural reward blocking learning (check `mean_structural_reward` — if < 0.3 after 100 steps, the format prompt needs adjustment)
4. `Δ_O/S` attack active (model gaming format, not learning reasoning)
5. Domain curriculum broken (check per-domain `outcome_reward` in W&B)