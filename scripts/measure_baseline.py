# scripts/measure_baseline.py

import argparse
import json
import torch
from pathlib import Path
from eval.benchmark_eval import BenchmarkEvaluator, BENCHMARKS


def _build_engine(model_id: str, device: str):
    """
    Device-aware engine builder.
    - cuda  → KaggleRolloutEngine (BitsAndBytes 4-bit for 3B on P100)
    - mps   → M4RolloutEngine (HF generate + MPS)
    - cpu   → M4RolloutEngine with CPU fallback
    """
    if device == "cuda":
        from kaggle.kaggle_rollout_engine import build_kaggle_engine
        print(f"Using KaggleRolloutEngine (4-bit BnB CUDA) for {model_id}")
        return build_kaggle_engine(model_id, load_in_4bit=True)
    else:
        from m4.m4_rollout_engine import build_m4_engine
        print(f"Using M4RolloutEngine ({device}) for {model_id}")
        return build_m4_engine(model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "strategyqa"])
    parser.add_argument("--output",     default="reports/actual_baselines.json")
    parser.add_argument(
        "--baseline_path",
        default=None,
        help="Path to existing baselines JSON for delta calculation",
    )
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"--- MEASURING ACTUAL BASELINES FOR {args.model} ---")
    print(f"Samples per benchmark: {args.n_samples}")
    print(f"Using device: {device}")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {total:.1f} GB  |  pre-load allocated: {allocated:.2f} GB")

    engine = _build_engine(args.model, device)

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM post-load: {allocated:.2f} GB")

    evaluator = BenchmarkEvaluator(generate_fn=engine.generate_for_eval)

    # Load existing baselines for delta calculation if provided
    existing = {}
    if args.baseline_path and Path(args.baseline_path).exists():
        with open(args.baseline_path) as f:
            data = json.load(f)
            existing = data.get("benchmarks", {})

    results = {
        "model_id":             args.model,
        "device":               device,
        "n_samples_per_bench":  args.n_samples,
        "benchmarks":           {},
    }

    for bench in args.benchmarks:
        print(f"\nEvaluating {bench}...")
        r = evaluator.run_benchmark(bench, greedy_only=True, n_samples=args.n_samples)
        results["benchmarks"][bench] = r

        delta_str = ""
        if bench in existing:
            prev = existing[bench].get("greedy_acc", 0.0)
            delta = r["greedy_acc"] - prev
            delta_str = f"  delta vs prev baseline: {delta:+.4f}"

        print(f"  Result: {r['greedy_acc']:.4f}  (lit baseline: {r['baseline_lit']:.4f}){delta_str}")

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Baseline measurements complete. Saved to {args.output}")


if __name__ == "__main__":
    main()