import argparse, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from eval.benchmark_eval import BenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="outputs/final")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--output", default="reports/final_eval.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading base model Qwen/Qwen2.5-3B-Instruct on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16 if device=="mps" else torch.float32,
    ).to(device)
    
    print(f"Applying adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    from m4.m4_rollout_engine import M4RolloutEngine
    engine = M4RolloutEngine(model, tokenizer, device=device)
    evaluator = BenchmarkEvaluator(generate_fn=engine.generate_for_eval)

    results = {
        "model_id": "Qwen/Qwen2.5-3B-Instruct + " + args.adapter_path,
        "device": device,
        "n_samples_per_bench": args.n_samples,
        "benchmarks": {}
    }

    for bench in ["gsm8k", "mmlu", "strategyqa"]:
        print(f"Evaluating {bench}...")
        r = evaluator.run_benchmark(bench, greedy_only=True, n_samples=args.n_samples)
        results["benchmarks"][bench] = r
        print(f"  Accuracy: {r['greedy_acc']:.4f}")

    out_path = args.output
    Path(out_path).parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
