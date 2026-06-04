with open('README.md', 'r') as f:
    content = f.read()

old_table = """| Benchmark | Baseline (Measured) | Post-RL | Delta | Lit Baseline | Target Met |
|-----------|--------------------|---------|-------|--------------|------------|
| **GSM8K** | 0.5000 | 0.5500 | +0.0500 | 0.4500 | YES |
| **MMLU** | 0.3000 | 0.6000 | +0.3000 | 0.5000 | YES |
| **STRATEGYQA** | 0.9000 | 0.7000 | -0.2000 | 0.6000 | YES |

> **Summary:** All KPI targets were successfully met. The model learned to correctly wrap reasoning steps within the `<think>` tags and successfully navigated the UCB multi-armed bandit curriculum without domain collapse."""

new_table = """| Benchmark | Baseline (Measured) | Post-RL | Delta | Lit Baseline | Target Met |
|-----------|--------------------|---------|-------|--------------|------------|
| **GSM8K** | 0.5000 | 0.5500 | +0.0500 | 0.4500 | YES |
| **MMLU** | 0.3000 | 0.6000 | +0.3000 | 0.5000 | YES |
| **STRATEGYQA** | 0.9000 | 0.7000 | -0.2000 | 0.6000 | NO — REGRESSION |

> **Summary:** The model successfully improved on GSM8K and MMLU, but experienced catastrophic forgetting on StrategyQA. Because StrategyQA started with a high baseline (0.900), the UCB scheduler likely under-sampled it compared to the weaker domains, allowing gradient pressure from MMLU/GSM8K to overwrite StrategyQA's internal representations."""

content = content.replace(old_table, new_table)

with open('README.md', 'w') as f:
    f.write(content)
