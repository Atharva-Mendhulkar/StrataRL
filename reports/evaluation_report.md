# StrataRL EXP_01 Evaluation Report

- **Model**: Qwen/Qwen2.5-3B-Instruct + outputs/final
- **Device**: mps
- **Baseline samples**: 20
- **Eval samples**: 20

## Results

| Benchmark | Baseline (Measured) | Post-RL | Delta | Lit Baseline | Target Met |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | 0.5000 | 0.5500 | +0.0500 | 0.4500 | YES |
| **MMLU** | 0.3000 | 0.6000 | +0.3000 | 0.5000 | YES |
| **STRATEGYQA** | 0.9000 | 0.7000 | -0.2000 | 0.6000 | YES |

## Summary

All KPI targets met: YES
