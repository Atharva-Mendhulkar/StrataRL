# scripts/generate_report.py
"""
Compares pre-training baselines against post-training evaluation results
and prints a structured markdown report.
"""

import argparse
import json
from pathlib import Path


def generate_report(baseline_path: str, result_path: str) -> str:
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    with open(result_path) as f:
        result_data = json.load(f)

    baseline_benches = baseline_data.get("benchmarks", {})
    result_benches   = result_data.get("benchmarks", {})

    model_id = result_data.get("model_id", "unknown")
    device   = result_data.get("device", "unknown")

    lines = [
        "# StrataRL EXP_01 Evaluation Report",
        "",
        f"- **Model**: {model_id}",
        f"- **Device**: {device}",
        f"- **Baseline samples**: {baseline_data.get('n_samples_per_bench', '?')}",
        f"- **Eval samples**: {result_data.get('n_samples_per_bench', '?')}",
        "",
        "## Results",
        "",
        "| Benchmark | Baseline (Measured) | Post-RL | Delta | Lit Baseline | Target Met |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    all_targets_met = True
    for bench in ["gsm8k", "mmlu", "strategyqa"]:
        b_acc  = baseline_benches.get(bench, {}).get("greedy_acc", 0.0)
        r_acc  = result_benches.get(bench, {}).get("greedy_acc", 0.0)
        lit    = result_benches.get(bench, {}).get("baseline_lit", 0.0)
        target = result_benches.get(bench, {}).get("target_met", False)
        delta  = r_acc - b_acc
        sign   = "+" if delta >= 0 else ""
        met    = "YES" if target else "NO"
        if not target:
            all_targets_met = False
        lines.append(
            f"| **{bench.upper()}** | {b_acc:.4f} | {r_acc:.4f} | "
            f"{sign}{delta:.4f} | {lit:.4f} | {met} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"All KPI targets met: {'YES' if all_targets_met else 'NO'}",
        "",
    ]

    if not all_targets_met:
        lines += [
            "> One or more benchmarks did not meet the +10% improvement target.",
            "> Review W&B for domain-specific KL drift and Δ_O/S metrics.",
            "",
        ]

    report = "\n".join(lines)
    print(report)

    report_path = Path("reports/evaluation_report.md")
    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text(report)
    print(f"\n✓ Report saved to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to actual_baselines.json")
    parser.add_argument("--result",   required=True, help="Path to final_eval.json")
    args = parser.parse_args()
    generate_report(args.baseline, args.result)
