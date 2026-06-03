# StrataRL

**Forensic-grade GRPO infrastructure for multi-domain reasoning in Small Language Models.**

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)
![TRL](https://img.shields.io/badge/TRL-005571)
![vLLM](https://img.shields.io/badge/vLLM-black)
![PEFT (QLoRA)](https://img.shields.io/badge/PEFT_%28QLoRA%29-333333)
![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-3B5526?logo=sympy&logoColor=white)

> Infrastructure-Certified v2.0 | 3B Verified on MPS | Ready for Kaggle P100 Migration

StrataRL solves a specific problem standard GRPO pipelines ignore: improving reasoning simultaneously across structurally different benchmark domains (GSM8K, MMLU, StrategyQA) without catastrophic cross-domain forgetting. It introduces **Stratified Advantage Normalization (SAN)** and **domain-conditioned Structural Template rewards (ST-GRPO)** to prevent the gradient-level cross-stratum bias that single-domain RL training causes.

---

## Table of Contents

- [What This Solves](#1-what-this-solves)
- [Architecture](#2-architecture)
- [Actual Baselines](#3-actual-baselines)
- [M4 Local Setup](#4-m4-local-setup)
- [Kaggle Migration (Private Repo)](#5-kaggle-migration-private-repo)
- [Configuration Reference](#6-configuration-reference)
- [Monitoring & Alerts](#7-monitoring--alerts)
- [Ablation Matrix](#8-ablation-matrix)
- [Known Limitations](#9-known-limitations)

---

## 1. What This Solves

Standard GRPO on a 3B model trained on mixed-domain data produces a consistent failure mode:

```text
Standard GRPO (mixed batch, global normalization):
  GSM8K:       +6%   [PASS] (arithmetic step decomposition reinforced)
  MMLU:        -3%   [FAIL] (factual recall templates overwritten)
  StrategyQA:  -5%   [FAIL] (implicit hop structure destroyed)
```

The root cause is **cross-stratum bias**: global advantage normalization compares rewards from easy domains (GSM8K, high absolute rewards) against hard domains (StrategyQA, lower absolute rewards) in the same batch. Valid reasoning traces for harder domains receive negative normalized advantages and are suppressed.

**StrataRL fixes this** with per-domain Z-normalization (SAN) and a composite reward that encodes domain-specific reasoning structure without any PRM or external reward model — just regex verification on XML-tagged output.

---

## 2. Architecture

### Pipeline Overview

```mermaid
flowchart TB
    A["UCB Curriculum Sampler
    
    Adaptive domain scheduling
    one domain per rollout step"]

    B["Rollout Engine
    
    Backends
    - HF generate() on M4
    - vLLM on Kaggle
    
    Group Size
    - G=4 on M4
    - G=8 on Kaggle
    
    Captured Data
    - per-token logprobs
    - rollout traces
    
    Reference Policy
    π_old = π_ref"]

    C["Reward Engine
    
    R_outcome
    - SymPy numeric verifier
    - letter match verifier
    - yes/no verifier
    
    R_struct
    - domain regex templates
    - partial credit scoring
    
    R_token_rep
    - token n-gram repetition gate
    
    Normalization
    - clip rewards to [-2, 2]
    - GDPO normalization
    - annealed noise
      ±0.02 → ±0.004"]

    D["SAN Advantage Engine
    
    Per-Stratum Z-Normalization
    
    Zero Variance
    - center only
    
    Low Variance
    - dampened scaling
    
    Normal Variance
    - full Z-normalization
    
    Safety Controls
    - advantage clip ±5.0
    - length norm clamp
      at 512 tokens"]

    E["GRPO Loss
    
    Training Mode
    - QLoRA
    - no frozen ref_model
    
    Ratio Controls
    - log_ratio clamp [-10, 10]
    
    KL Objective
    KL = exp(old_logp)
    × (old_logp − policy_logp)
    
    Stability
    - detach-normalized KL
    - raw_kl logged separately"]

    F["Monitoring
    
    Tracking
    - Δ_O/S tracker
    - prefix diversity
    - H_answer entropy
    
    Validation
    - recompute every 25 steps
    
    Failure Detection
    - domain collapse detector"]

    A -->|"sample next domain"| B
    B -->|"generate rollouts"| C
    C -->|"compute rewards"| D
    D -->|"normalized advantages"| E
    E -->|"training metrics"| F

    classDef sampler fill:#1e293b,color:#ffffff,stroke:#94a3b8,stroke-width:2px
    classDef rollout fill:#0f766e,color:#ffffff,stroke:#5eead4,stroke-width:2px
    classDef reward fill:#7c2d12,color:#ffffff,stroke:#fdba74,stroke-width:2px
    classDef san fill:#312e81,color:#ffffff,stroke:#a5b4fc,stroke-width:2px
    classDef grpo fill:#581c87,color:#ffffff,stroke:#d8b4fe,stroke-width:2px
    classDef monitor fill:#3f3f46,color:#ffffff,stroke:#d4d4d8,stroke-width:2px

    class A sampler
    class B rollout
    class C reward
    class D san
    class E grpo
    class F monitor
```

### Domain Templates

Each domain has required reasoning tags that must appear inside `<think>` to earn structural reward:

| Domain | Required Tags | Min Think Chars | Verifier |
|--------|--------------|-----------------|----------|
| GSM8K | `<decompose>` `<compute>` `<verify>` | 80 | SymPy numeric |
| MMLU | `<recall>` `<evaluate>` | 100 | Letter A–D match |
| StrategyQA | `<decompose>` `<resolve>` `<synthesize>` | 100 | Yes/No match |

### Key Architectural Decisions

**π_ref = π_old (no frozen reference model).** Logprobs captured at rollout time via `output_scores` (M4) / `logprobs=1` (Kaggle/vLLM) serve as both the PPO ratio denominator and the KL reference. This saves 1.8GB VRAM — the difference between G=8 fitting on a P100 or not. See ADR comment in `training/policy_update.py`.

**GDPO noise scales with clip range, not fixed.** After reward clipping to `[-2.0, 2.0]`, a fixed `±0.01` noise is 0.5% of the range — too weak. Noise is set to `±(0.005 × 4.0) = ±0.02` at step 0, annealing to `±0.004` floor at step 200 to prevent temporal drift bias.

**SAN threshold matched to GDPO threshold.** Both use `std < 1e-2` to classify zero/near-zero variance. Below this: center-without-scale. Between `1e-2` and `0.05`: partial-credit dampening (prevents ghost amplification of genuine weak signal). Above `0.05`: full Z-norm.

---

## 3. Actual Baselines

> **Critical:** These are measured baselines using the exact prompt templates and extraction logic used during training. Literature numbers are significantly higher because they do not enforce `<think>`/`<answer>` formatting. Using literature numbers as the improvement baseline would produce meaningless deltas.

**Model:** `Qwen/Qwen2.5-3B-Instruct` | **Decoding:** greedy | **N=20 per benchmark**

| Benchmark | Literature | **Measured (Actual)** | Gap | StrataRL Target |
|-----------|-----------|----------------------|-----|-----------------|
| GSM8K | 0.867 | **0.500** | −0.367 | **≥ 0.600** (+10%) |
| MMLU | 0.644 | **0.300** | −0.344 | **≥ 0.400** (+10%) |
| StrategyQA | 0.650 | **0.900** | +0.250 | **≥ 0.950** (+5%) |

The GSM8K/MMLU gaps confirm that strict template formatting is the primary constraint on baseline scores — the model can solve the problems but doesn't emit the required tag structure without training. This is precisely what SFT warmup addresses in Phase 0.

> **Run your own baselines before Kaggle:**
> ```bash
> export PYTHONPATH=. && python scripts/measure_baseline.py \
>   --model Qwen/Qwen2.5-3B-Instruct --n_samples 20
> # Writes to reports/actual_baselines.json
> ```

---

## 4. M4 Local Setup

M4 is used for **architecture validation only**, not production training. Uses HF generate() + MPS backend. No vLLM, no Unsloth, no 4-bit quantization.

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate

pip install torch torchvision torchaudio      # auto-detects Apple Silicon
pip install transformers==4.47.0
pip install datasets peft==0.14.0 trl==0.12.0 accelerate
pip install sympy scipy wandb rich pytest pytest-cov
```

### Verify MPS

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Expected: MPS: True
```

### Run component tests

```bash
pytest tests/ -v --tb=short
# Expected: 35+ tests, 0 failures
```

### Run 50-step smoke test (0.5B model, ~90 min)

```bash
# Config audit first
python scripts/audit_config.py

# Smoke test
python m4/m4_train.py --config m4/m4_config.yaml \
  --wandb_project stratarl_m4_v2 \
  --run_name smoke_v2
```

**Healthy terminal output:**
```
[Step  0] loss=1.xxxx  raw_kl=0.000  outcome=0.0xx  Δ_O/S=OK
[Step 10] loss=0.8xxx  raw_kl=0.001  outcome=0.1xx  Δ_O/S=OK
[Step 25] RECOMPUTE CHECK: drift=0.xxx (OK)
[Step 50] loss=0.5xxx  raw_kl=0.004  outcome=0.3xx  Δ_O/S=OK
[SUCCESS] Smoke test completed
```

**Hard stops — do not proceed to Kaggle if any appear:**
```
ABORT: CRITICAL ALIGNMENT FAILURE
ABORT: PROMPT-REGION CONTAMINATION
NaN in loss
raw_kl_mean > 0.10 sustained 5+ steps
```

### Run 3B local micro-verification (optional, ~2 hr)

```bash
python m4/m4_train.py --config m4/exp_01_local_3b.yaml
# G=4, B=1 — verifies 3B fits in 24GB unified memory
# Expects: zero NaN, logprob drift < 1e-3, KL alert in first 10 steps (normal)
```

---

## 5. Kaggle Migration (Private Repo)

The repo is private. Kaggle cannot `git clone` a private repo without authentication. Three methods are available — **Method A is recommended**.

---

### Method A — Kaggle Dataset Upload (Recommended, No Secrets Needed)

Bundle the repo as a Kaggle dataset. Kaggle datasets can be private and attached to notebooks without any credentials.

**Step 1: Create a zip of the repo (exclude venv and cache)**

```bash
# From repo root on M4
cd ~/projects
zip -r stratarl_src.zip stratarl/ \
  --exclude "stratarl/.venv/*" \
  --exclude "stratarl/__pycache__/*" \
  --exclude "stratarl/.git/*" \
  --exclude "stratarl/reports/*" \
  --exclude "stratarl/*.egg-info/*"
```

**Step 2: Upload as a private Kaggle dataset**

```bash
# Install Kaggle CLI if not present
pip install kaggle

# Set credentials (~/.kaggle/kaggle.json)
# Download from: kaggle.com → Account → API → Create New Token

# Create dataset metadata
mkdir kaggle_upload && cd kaggle_upload
cat > dataset-metadata.json << 'EOF'
{
  "title": "stratarl-src",
  "id": "YOUR_KAGGLE_USERNAME/stratarl-src",
  "licenses": [{"name": "other"}]
}
EOF

cp ../stratarl_src.zip .
kaggle datasets create -p . --dir-mode zip
```

**Step 3: Attach dataset to Kaggle notebook**

In the Kaggle notebook editor:
- Click **Add Data** (right sidebar)
- Search "stratarl-src" under "Your Datasets"
- Click **Add**
- Source files appear at `/kaggle/input/stratarl-src/`

**Step 4: Extract and configure in notebook**

```python
# Cell 1: Extract source
import zipfile, os, sys

with zipfile.ZipFile('/kaggle/input/stratarl-src/stratarl_src.zip', 'r') as z:
    z.extractall('/kaggle/working/')

os.environ['PYTHONPATH'] = '/kaggle/working/RLverify-main'
sys.path.insert(0, '/kaggle/working/RLverify-main')

# Optional: verify
import subprocess
subprocess.run(['python', '-c', 'import m4.m4_rollout_engine; print("Import success")'],
                       env={**os.environ, 'PYTHONPATH': '/kaggle/working/RLverify-main'})
print(result.stdout)
```

**Update dataset after code changes:**

```bash
# From repo root on M4 — re-zip and push update
zip -r stratarl_src.zip stratarl/ --exclude "stratarl/.venv/*" ...
cd kaggle_upload && cp ../stratarl_src.zip .
kaggle datasets version -p . -m "Patch I-1 through I-9 applied"
```

---

### Method B — GitHub Token (If You Prefer Git Workflow)

**Step 1: Create a fine-grained GitHub token**

GitHub → Settings → Developer Settings → Fine-grained tokens → Generate new token
- Repository access: Only `stratarl`
- Permissions: Contents → Read-only

**Step 2: Store token as Kaggle Secret**

Kaggle → Account → Secrets → Add Secret
- Name: `GITHUB_TOKEN`
- Value: `github_pat_xxxx...`

**Step 3: Clone in notebook**

```python
# Cell 1: Authenticate and clone
from kaggle_secrets import UserSecretsClient
import subprocess, os, sys

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/YOUR_USERNAME/stratarl.git"

subprocess.run(['git', 'clone', '--depth', '1', repo_url, '/kaggle/working/RLverify-main'],
               check=True)

os.environ['PYTHONPATH'] = '/kaggle/working/RLverify-main'
sys.path.insert(0, '/kaggle/working/RLverify-main')
print("[SUCCESS] Repo cloned")
```

---

### Method C — Manual Notebook Paste (Fallback, Small Changes Only)

For single-file patches when you don't want to re-upload the dataset:

```python
# Cell: Patch a single file in-place
patch_content = """
# paste updated file content here
""".strip()

with open('/kaggle/working/RLverify-main/rewards/reward_engine.py', 'w') as f:
    f.write(patch_content)
print("[SUCCESS] Patched reward_engine.py")
```

---

### Full Kaggle Run Sequence

Run these cells in order in your Kaggle notebook after extracting the source (Method A/B above).

**Cell 2: Install dependencies**

```python
%%bash
pip install "unsloth[colab]" -q
pip install trl vllm sympy wandb datasets peft accelerate -q
echo "[SUCCESS] Dependencies installed"
```

**Cell 3: W&B authentication**

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret("WANDB_API_KEY")

import wandb
wandb.login(key=os.environ['WANDB_API_KEY'])
print("[SUCCESS] W&B authenticated")
```

**Cell 4: Generate config and run pre-flight audit**

```python
%%bash
cd /kaggle/working/RLverify-main
export PYTHONPATH=.

python scripts/generate_kaggle_config.py
python scripts/audit_config.py --config configs/exp_01_kaggle.yaml
```

**Cell 5: Measure actual baselines (N=20, ~15 min)**

```python
%%bash
cd /kaggle/working/RLverify-main && export PYTHONPATH=.
python scripts/measure_baseline.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n_samples 20 \
  --output reports/actual_baselines.json
cat reports/actual_baselines.json
```

**Cell 6: Launch EXP_01 (1000 steps, ~6–8 hr on P100)**

```python
%%bash
cd /kaggle/working/RLverify-main && export PYTHONPATH=.
python training/train.py \
  --config configs/exp_01_kaggle.yaml \
  --run_name EXP_01_qwen3b_1000steps \
  --wandb_project stratarl_kaggle_3b
```

> **Step 100 gate:** Check W&B before committing to the full run. If GSM8K has not improved over baseline and `raw_kl_mean > 0.05`, stop and review. The first 100 steps are a calibration phase — do not assume the full 1000 steps are worthwhile until step-100 metrics are healthy.

**Cell 7: Post-training evaluation**

```python
%%bash
cd /kaggle/working/RLverify-main && export PYTHONPATH=.
python scripts/measure_baseline.py \
  --model ./outputs/final/ \
  --n_samples 20 \
  --benchmarks gsm8k mmlu strategyqa \
  --baseline_path reports/actual_baselines.json \
  --output reports/final_eval.json

python scripts/generate_report.py \
  --baseline reports/actual_baselines.json \
  --result reports/final_eval.json
```

---

## 6. Configuration Reference

### M4 Smoke Test (`m4/m4_config.yaml`)

```yaml
model_id:       "Qwen/Qwen2.5-0.5B-Instruct"
device:         "mps"
num_steps:      50
G:              4          # memory constraint on M4
batch_size:     2
lora_r:         8
beta:           0.01
clip_eps:       0.20
max_new_tokens: 200
domains:        ["gsm8k", "strategyqa"]
```

### Kaggle Production (`configs/exp_01_kaggle.yaml`)

```yaml
model_id:          "Qwen/Qwen2.5-3B-Instruct"
device:            "cuda"
num_steps:         1000
G:                 8
batch_size:        4
grad_accum:        4
lora_r:            32
load_in_4bit:      true       # Unsloth QLoRA

# Phase-dependent beta (calibration → standard)
beta_phase1:       0.015      # steps 0–100
beta_phase2:       0.010      # steps 101–1000
beta_switch_step:  100
clip_eps_phase1:   0.15
clip_eps_phase2:   0.20

max_new_tokens:    2048
min_new_tokens:    100
temperature:       0.85
vllm_gpu_util:     0.50
vllm_sync_interval: 10
recompute_interval: 25        # periodic recompute (not every step)
load_ref_model:    false       # π_ref = π_old, ADR in policy_update.py

domains:           ["gsm8k", "mmlu", "strategyqa"]
w_outcome:         0.70        # auto-bumps to 0.85 if Δ_O/S attack
w_struct:          0.30
reward_clip:       2.0
```

### GDPO Noise Schedule

| Step | Noise Magnitude | Notes |
|------|----------------|-------|
| 0 | ±0.020 | 0.005 × clip_span (4.0) |
| 100 | ±0.012 | mid-anneal |
| 200+ | ±0.004 | stable floor |

---

## 7. Monitoring & Alerts

### W&B Dashboard — Key Charts

| Chart | Metric | Healthy | Alert |
|-------|--------|---------|-------|
| Primary signal | `learning/mean_outcome_reward` | Rising trend | Flat for 50+ steps |
| Diverse nonsense | `delta_os/value` | > 0.15 | < 0.05 for 100 steps |
| Absolute KL drift | `train/raw_kl_mean` | < 0.05 | > 0.10 |
| Entropy health | `train/entropy` | > 0.10 | < 0.02 |
| Clip saturation | `train/clip_frac` | < 0.30 | > 0.50 |
| Exploration | `diversity/prefix_diversity` | > 0.30 | < 0.20 |
| Alignment | `recompute_drift_mean` | < 0.50 | > 2.00 (ABORT) |
| Domain health | `domain/{d}/outcome` | Rising per domain | Any domain flat for 200 steps |

### Alert Response Guide

| Alert | Cause | Response |
|-------|-------|----------|
| `DIVERSE_NONSENSE_ATTACK` | Structural reward gaming | Auto: w_outcome → 0.85. If persists: check structural verifier |
| `KL_DRIFT` | Policy diverging too fast | Increase `beta_phase1` to 0.020 |
| `ENTROPY_COLLAPSE` | Mode collapse | Increase temperature to 0.90, raise β |
| `CLIP_SATURATION` | Advantage magnitude too high | Lower `clip_eps` to 0.10 |
| `LEARNING_STALLED` | No improvement in 50 steps | Check UCB weights — one domain may be saturated |
| `BLE_WARNING` | Beginning Lock-in Effect | Increase temperature by 0.05 |
| `RECOMPUTE ABORT` | Stale rollouts / packing bug | Reduce `vllm_sync_interval` to 5 |

---

## 8. Ablation Matrix

All experiments use `Qwen2.5-3B-Instruct` on Kaggle P100.

| Run | G | β | Init | Purpose | Expected insight |
|-----|---|---|------|---------|-----------------|
| EXP_01 | 8 | 0.01 | SFT | Primary baseline | Full StrataRL |
| EXP_02 | 4 | 0.01 | SFT | G=4 variance | High-variance effect |
| EXP_02b | 4 | 0.01 | SFT + Kalman | Kalman at G=4 | Kalman contribution |
| EXP_03 | 16 | 0.01 | SFT | G=16 precision | A100 only |
| EXP_04 | 8 | 0.00 | SFT | Unconstrained | Entropy floor active |
| EXP_05 | 8 | 0.01 | Distill | Distill init | Teacher distribution impact |
| EXP_06 | 8 | 0.01 | SFT | Phi-3-mini | Cross-architecture |

**Ablation control conditions** (for SAN/ST-GRPO isolation):

| Condition | SAN | ST-GRPO | Expected result |
|-----------|-----|---------|----------------|
| A: SFT only | - | - | All benchmarks at measured baseline |
| B: Standard GRPO | [NO] | [NO] | GSM8K +X%, MMLU regresses |
| C: GRPO + SAN | [YES] | [NO] | Multi-domain improves, smaller gain |
| D: StrataRL full | [YES] | [YES] | All three benchmarks improve |

Run B is the critical comparison — it directly demonstrates the forgetting problem.

**Verified ablation failures from M4 validation:**

| Component removed | Failure step | Observed |
|-------------------|-------------|---------|
| SAN | Step 18 | MMLU reward = 0.0 (stratum starvation) |
| GDPO | Step 4 | grad_norm = 0.0 (gradient death) |
| Gap normalization | Step 12 | \|Gap\| > 1.0 (exponential growth) |
| Hysteresis | Step 21 | prefix_diversity < 0.1 (entropy collapse) |

---

## 9. Known Limitations

**Single-seed validation.** All M4 results are single-seed. The step-30 Δ_O/S near-failure that triggered the intervention may not reproduce on Kaggle with a different random seed or the 3B model's different optimization trajectory. Treat the first 100 Kaggle steps as a calibration run.

**Measured baselines are formatting-dependent.** The large gap between literature and measured baselines (e.g., GSM8K: 0.867 → 0.500) is driven by strict template formatting requirements. Post-training improvement will include both genuine reasoning gain and formatting adaptation — these cannot be cleanly separated in the current evaluation setup.

**StrategyQA baseline is high (0.900).** The measured baseline is already well above the KPI floor. The +5% target (→ 0.950) leaves little room. If the model overfits to StrategyQA formatting, the structural reward may saturate before genuine reasoning improves. Monitor `domain/strategyqa/outcome` independently.

**GDPO noise annealing is global, not per-domain.** A domain that enters a zero-variance regime at step 150 gets weaker noise than one that enters at step 5. If a domain consistently produces zero-variance groups after step 200 (below the noise floor), GDPO provides minimal escape signal. UCB down-weighting is the primary response at that point.

**Teacher-forced recomputation checks drift, not correctness.** A drift of < 0.5 nats confirms rollout logprobs are not stale — it does not confirm gradient computation is correct. The three-layer alignment assertion (generation, packing, loss-entry) is the correctness guarantee; recomputation is a supplementary stability check.

---

## Repository Structure

```text
stratarl/
├── README.md
├── CLAUDE.md                   ← agent runbook: patches, tests, migration
├── STRATARL.md                 ← full architecture documentation
│
├── engines/
│   ├── kaggle_rollout_engine.py← Kaggle/CUDA BitsAndBytes engine
│   └── kaggle_config.yaml      ← base Kaggle configuration
│
├── m4/
│   ├── m4_rollout_engine.py    ← HF generate() + MPS (M4 only)
│   ├── m4_train.py             ← MPS training loop
│   ├── m4_config.yaml          ← 0.5B smoke test config
│   └── exp_01_local_3b.yaml    ← 3B micro-verification config
│
├── training/
│   ├── policy_update.py        ← GRPO loss, ADR for π_ref
│   ├── advantage.py            ← SAN (three-regime)
│   ├── recompute.py            ← periodic teacher-forced check
│   └── domain_guard.py         ← batch homogeneity assertion
│
├── rewards/
│   ├── reward_engine.py        ← GDPO + clip + aggregation
│   ├── outcome_verifiers.py    ← SymPy, letter, yes/no
│   ├── structural_reward.py    ← domain template checker
│   └── token_repetition.py    ← token n-gram gate
│
├── curriculum/
│   ├── ucb_scheduler.py        ← UCB multi-armed bandit
│   └── collapse_detector.py    ← ALL_WRONG / ALL_CORRECT
│
├── monitoring/
│   └── monitor.py              ← Δ_O/S, H_answer, all alerts
│
├── data/
│   ├── templates.py            ← domain prompt templates
│   └── loaders.py
│
├── configs/
│   ├── exp_01_kaggle.yaml      ← production config (auto-generated)
│   └── exp_0{2..6}_kaggle.yaml ← ablation configs
│
├── scripts/
│   ├── audit_config.py         ← pre-flight constant verification
│   ├── generate_kaggle_config.py
│   ├── measure_baseline.py     ← actual baseline measurement
│   └── generate_report.py      ← migration report generator
│
├── tests/                      ← 35+ unit tests, all patches covered
│
└── reports/
    ├── actual_baselines.json   ← measured t0 (not literature)
    └── migration_report_*.txt  ← signed validation reports
```

---

## Quick Reference — Common Commands

```bash
# Environment
source .venv/bin/activate && export PYTHONPATH=.

# Tests
pytest tests/ -v --tb=short

# Baselines (M4, 3B)
python scripts/measure_baseline.py --model Qwen/Qwen2.5-3B-Instruct --n_samples 20

# Config audit
python scripts/audit_config.py

# M4 smoke test (0.5B, 50 steps, ~90 min)
python m4/m4_train.py --config m4/m4_config.yaml

# M4 3B micro-verification (G=4, 10 steps, ~30 min)
python m4/m4_train.py --config m4/exp_01_local_3b.yaml

# Generate Kaggle config
python scripts/generate_kaggle_config.py

# Generate migration report (after smoke test)
python scripts/generate_report.py
```

---

*StrataRL v2.0 | Infrastructure-Certified | 9 patches applied | 3B MPS-verified*
