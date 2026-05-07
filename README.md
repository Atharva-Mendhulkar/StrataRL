# StrataRL: Research-Grade GRPO Infrastructure for Reasoning Models

## Project Overview
StrataRL is a specialized reinforcement learning infrastructure designed for training and certifying reasoning-capable language models. It implements Group Relative Policy Optimization (GRPO) with Stratified Advantage Normalization (SAN) to stabilize training across heterogeneous reasoning domains such as mathematics (GSM8K), general knowledge (MMLU), and logical reasoning (StrategyQA).

The system is hardened with a suite of forensic patches (I-1 through I-9) to prevent numerical instability, catastrophic alignment drift, and reward hacking common in long-horizon RLHF pipelines.

## System Architecture

### 1. Optimization Core: GRPO
StrataRL utilizes a reference-model-free GRPO implementation. By setting the reference policy to the old policy (pi_old) and using rollout-time logprobs, the system achieves a significant reduction in VRAM overhead (approximately 1.8GB for a 3B model) while maintaining the mathematical integrity of the KL penalty.

### 2. Stratified Advantage Normalization (SAN)
To prevent "domain starvation" where high-variance tasks dominate the gradient signal, SAN groups rollouts by domain. Advantages are normalized within each stratum, ensuring that progress in harder domains (e.g., mathematics) is not suppressed by easy wins in simpler tasks.

### 3. Forensic Hardening Layer
The infrastructure includes nine critical safety invariants:
- I-1: GDPO Normalization with annealed noise to prevent zero-variance collapse.
- I-2: Reference-free KL penalty for memory efficiency.
- I-3: Periodic teacher-forced recomputation to verify logprob alignment.
- I-4: Delta_O/S Tracker for detecting nonsense formatting attacks.
- I-5: Partial-credit dampening in SAN.
- I-7: Batch-domain homogeneity enforcement.
- I-8: Prompt-region logprob contamination assertions.
- I-9: Log-space ratio clamping to prevent exponential overflows.

## Directory Structure
- /training: Core RL logic, loss functions, and advantage calculation.
- /rewards: Structural and outcome reward engines.
- /monitoring: Forensic trackers and runtime invariant auditing.
- /eval: Multi-domain benchmark evaluators and extractors.
- /m4: Local M-series Mac verification environment.
- /configs: Production deployment manifests for 3B migration.
- /scripts: Utilities for baseline measurement and ablation studies.

## Setup and Installation

### 1. Environment Initialization
The project uses a local virtual environment. Note that the environment directory is named .venv.

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Audit
Before running any training, verify the infrastructure constants:
```bash
export PYTHONPATH=.
./.venv/bin/python scripts/audit_config.py --config m4/m4_config.yaml
```

## Running the Pipeline

### 1. Baseline Measurement
Establish actual performance metrics for the target model using the project's specific prompt templates:
```bash
export PYTHONPATH=.
./.venv/bin/python scripts/measure_baseline.py --model Qwen/Qwen2.5-3B-Instruct --n_samples 100
```

### 2. Control Experiment (EXP_01)
To run the full-scale 3B model training (500-1000 steps) on a CUDA-enabled production environment:
```bash
export PYTHONPATH=.
./.venv/bin/python m4/m4_train.py --config configs/exp_01_kaggle.yaml
```

### 3. Local Verification
To test the plumbing on an M-series Mac using a 0.5B model:
```bash
export PYTHONPATH=.
./.venv/bin/python m4/m4_train.py --config m4/m4_config.yaml
```

## Forensic Invariants and Alerts
The system monitors several runtime metrics to ensure training health:
- KL_DIVERGENCE_CRITICAL: Triggered if the policy drifts too far from the starting point within a single update.
- PROMPT_CONTAMINATION: Triggered if the training process attempts to modify logprobs in the fixed prompt region.
- NONSENSE_ATTACK_DETECTED: Triggered by the Delta_O/S tracker if structural rewards are maximized without corresponding outcome gains.
- RECOMPUTE_DRIFT: Triggered if the forward-pass logprobs differ from rollout-stored logprobs by more than 1e-3.
