# scripts/measure_baseline.py

import argparse, json, torch, os
from pathlib import Path
from eval.benchmark_eval import BenchmarkEvaluator, BENCHMARKS
from m4.m4_rollout_engine import build_m4_engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--benchmarks",  nargs="+", default=["gsm8k", "mmlu", "strategyqa"])
    parser.add_argument("--output",      default="reports/actual_baselines.json")
    args = parser.parse_args()

    print(f"--- MEASURING ACTUAL BASELINES FOR {args.model} ---")
    print(f"Samples per benchmark: {args.n_samples}")
    
    # Force MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build engine
    engine    = build_m4_engine(args.model)
    evaluator = BenchmarkEvaluator(generate_fn=engine.generate_for_eval)

    results = {
        "model_id": args.model,
        "n_samples_per_bench": args.n_samples,
        "benchmarks": {}
    }

    for bench in args.benchmarks:
        print(f"Evaluating {bench}...")
        r = evaluator.run_benchmark(bench, greedy_only=True, n_samples=args.n_samples)
        results["benchmarks"][bench] = r
        print(f"  Result: {r['greedy_acc']:.4f} (Lit baseline: {r['baseline_lit']:.4f})")

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Baseline measurements complete. Saved to {args.output}")

if __name__ == "__main__":
    main()
