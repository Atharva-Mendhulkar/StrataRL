# StrataRL M4 Development Guide

## Build Commands
- Run all tests: `pytest tests/ -v`
- Run reward tests: `pytest tests/test_reward_engine.py -v`
- Run SAN tests: `pytest tests/test_san_advantage.py -v`
- Run KL tests: `pytest tests/test_kl_computation.py -v`
- Run alignment tests: `pytest tests/test_alignment.py -v`

## Core Components
- `rewards/`: Reward engine and structural verifiers
- `training/`: Advantage computation and policy loss
- `curriculum/`: UCB domain scheduler
- `m4/`: M4-specific overrides for local validation

## Usage on M4
1. Source venv: `source .venv/bin/activate`
2. Run smoke test: `bash scripts/run_smoke_test.sh`

## Style Guidelines
- Use MPS for local acceleration.
- Do not use vLLM or Unsloth (unsupported on Apple Silicon).
- Maintain bfloat16 precision.
- Ensure KL formula is `p_old * (log_p_old - log_p_new)`.
