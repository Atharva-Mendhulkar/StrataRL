"""
Diagnostic evaluation: Base vs RL adapter on 20 GSM8K questions.
Saves per-sample predictions with full details for manual inspection.

Usage:
    python scripts/diagnose_adapter.py --adapter_path outputs/final --n_samples 20
"""

import argparse, json, torch, os, sys, time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from eval.benchmark_eval import (
    format_gsm8k_prompt, extract_answer, extract_think_length, verify_math, THINK_RE, ANSWER_RE
)


def load_model(device, adapter_path=None):
    """Load base model, optionally with RL adapter."""
    print(f"Loading Qwen/Qwen2.5-3B-Instruct on {device}...")

    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float32,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Applying RL adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, prompt, device, max_tokens=512):
    """Generate a single greedy completion and return full details."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    start = time.time()
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start

    comp_ids = outputs[0][prompt_len:]
    completion = tokenizer.decode(comp_ids, skip_special_tokens=True)
    n_tokens = len(comp_ids)

    return completion, n_tokens, elapsed


def diagnose_completion(completion, ground_truth):
    """Analyze a completion for common failure modes."""
    issues = []

    # Check for <think> tags
    has_think = bool(THINK_RE.search(completion))
    if not has_think:
        issues.append("MISSING_THINK_TAGS")

    # Check for <answer> tags
    has_answer = bool(ANSWER_RE.search(completion))
    if not has_answer:
        issues.append("MISSING_ANSWER_TAGS")

    # Check for malformed XML (opened but not closed)
    if "<think>" in completion.lower() and "</think>" not in completion.lower():
        issues.append("UNCLOSED_THINK_TAG")
    if "<answer>" in completion.lower() and "</answer>" not in completion.lower():
        issues.append("UNCLOSED_ANSWER_TAG")

    # Check for token repetition
    words = completion.split()
    if len(words) > 20:
        # Check for 4-gram repetition
        ngrams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
        from collections import Counter
        ngram_counts = Counter(ngrams)
        max_repeat = max(ngram_counts.values()) if ngram_counts else 0
        if max_repeat > 5:
            issues.append(f"TOKEN_REPETITION(max_4gram={max_repeat})")

    # Check if answer was extracted but wrong
    parsed = extract_answer(completion)
    if parsed and not verify_math(parsed, ground_truth):
        issues.append(f"WRONG_ANSWER(parsed='{parsed}', gt='{ground_truth}')")
    elif not parsed:
        issues.append("ANSWER_NOT_PARSEABLE")

    # Check think length
    think_len = extract_think_length(completion)
    if think_len < 5:
        issues.append(f"VERY_SHORT_REASONING(words={think_len})")

    # Check if completion is very short overall
    if len(words) < 10:
        issues.append(f"VERY_SHORT_COMPLETION(words={len(words)})")

    # Check if completion is truncated (hit max tokens without EOS)
    if len(words) > 400:
        issues.append(f"POSSIBLY_TRUNCATED(words={len(words)})")

    return issues


def run_diagnostic(adapter_path, n_samples, output_path, checkpoints=None):
    """Run base vs adapter comparison on GSM8K."""

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load GSM8K test set
    print("Loading GSM8K test set...")
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    print(f"Selected {len(ds)} samples\n")

    # Prepare questions
    samples = []
    for item in ds:
        prompt, gt = format_gsm8k_prompt(item)
        samples.append({
            "question": item["question"],
            "full_answer": item["answer"],
            "ground_truth": gt,
            "prompt": prompt,
        })

    # ── Evaluate base model ────────────────────────────────────────────
    print("=" * 60)
    print("EVALUATING: Base Qwen2.5-3B-Instruct (no adapter)")
    print("=" * 60)
    base_model, tokenizer = load_model(device, adapter_path=None)

    for i, s in enumerate(samples):
        comp, n_tok, elapsed = generate_one(base_model, tokenizer, s["prompt"], device)
        parsed = extract_answer(comp)
        correct = verify_math(parsed, s["ground_truth"]) if parsed else False
        issues = diagnose_completion(comp, s["ground_truth"])

        s["base_completion"] = comp
        s["base_parsed_answer"] = parsed
        s["base_correct"] = correct
        s["base_issues"] = issues
        s["base_tokens"] = n_tok
        s["base_time"] = round(elapsed, 2)
        s["base_think_len"] = extract_think_length(comp)

        status = "✓" if correct else "✗"
        print(f"  [{i+1}/{n_samples}] {status} parsed={parsed!r:<10} gt={s['ground_truth']:<10} "
              f"tokens={n_tok:<4} issues={issues or 'none'}")

    base_acc = sum(1 for s in samples if s["base_correct"]) / len(samples)
    print(f"\nBase model accuracy: {base_acc:.1%} ({sum(1 for s in samples if s['base_correct'])}/{len(samples)})\n")

    # Free base model
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Evaluate RL adapter ────────────────────────────────────────────
    # Build list of adapters to evaluate
    adapters_to_eval = [("rl_final", adapter_path)]
    if checkpoints:
        for ckpt in checkpoints:
            if Path(ckpt).exists():
                name = Path(ckpt).name
                adapters_to_eval.append((f"rl_{name}", ckpt))

    for adapter_name, adapter_dir in adapters_to_eval:
        print("=" * 60)
        print(f"EVALUATING: {adapter_name} ({adapter_dir})")
        print("=" * 60)
        rl_model, tokenizer = load_model(device, adapter_path=adapter_dir)

        for i, s in enumerate(samples):
            comp, n_tok, elapsed = generate_one(rl_model, tokenizer, s["prompt"], device)
            parsed = extract_answer(comp)
            correct = verify_math(parsed, s["ground_truth"]) if parsed else False
            issues = diagnose_completion(comp, s["ground_truth"])

            s[f"{adapter_name}_completion"] = comp
            s[f"{adapter_name}_parsed_answer"] = parsed
            s[f"{adapter_name}_correct"] = correct
            s[f"{adapter_name}_issues"] = issues
            s[f"{adapter_name}_tokens"] = n_tok
            s[f"{adapter_name}_time"] = round(elapsed, 2)
            s[f"{adapter_name}_think_len"] = extract_think_length(comp)

            status = "✓" if correct else "✗"
            print(f"  [{i+1}/{n_samples}] {status} parsed={parsed!r:<10} gt={s['ground_truth']:<10} "
                  f"tokens={n_tok:<4} issues={issues or 'none'}")

        adapter_acc = sum(1 for s in samples if s[f"{adapter_name}_correct"]) / len(samples)
        print(f"\n{adapter_name} accuracy: {adapter_acc:.1%} "
              f"({sum(1 for s in samples if s[f'{adapter_name}_correct'])}/{len(samples)})\n")

        del rl_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    base_correct = sum(1 for s in samples if s["base_correct"])
    print(f"  Base:     {base_correct}/{n_samples} ({base_correct/n_samples:.1%})")

    for adapter_name, _ in adapters_to_eval:
        rl_correct = sum(1 for s in samples if s[f"{adapter_name}_correct"])
        print(f"  {adapter_name}: {rl_correct}/{n_samples} ({rl_correct/n_samples:.1%})")

    # Per-sample delta
    adapter_name = adapters_to_eval[0][0]  # primary adapter
    print(f"\nPer-sample comparison (base vs {adapter_name}):")
    regressions = []
    improvements = []
    for i, s in enumerate(samples):
        base_c = s["base_correct"]
        rl_c = s[f"{adapter_name}_correct"]
        if base_c and not rl_c:
            regressions.append(i)
            print(f"  [{i+1}] REGRESSION: base ✓ → rl ✗ | gt={s['ground_truth']} | "
                  f"base_parsed={s['base_parsed_answer']} | rl_parsed={s[f'{adapter_name}_parsed_answer']} | "
                  f"rl_issues={s[f'{adapter_name}_issues']}")
        elif not base_c and rl_c:
            improvements.append(i)
            print(f"  [{i+1}] IMPROVEMENT: base ✗ → rl ✓ | gt={s['ground_truth']}")

    print(f"\n  Regressions: {len(regressions)}")
    print(f"  Improvements: {len(improvements)}")
    print(f"  Net delta: {len(improvements) - len(regressions):+d}")

    # ── Issue frequency ────────────────────────────────────────────────
    print(f"\nIssue frequency ({adapter_name}):")
    from collections import Counter
    all_issues = []
    for s in samples:
        for issue in s[f"{adapter_name}_issues"]:
            # Normalize parameterized issues
            issue_type = issue.split("(")[0]
            all_issues.append(issue_type)
    for issue, count in Counter(all_issues).most_common():
        print(f"  {issue}: {count}/{n_samples}")

    # ── Save predictions ───────────────────────────────────────────────
    # Remove prompt from saved output to keep file readable
    for s in samples:
        del s["prompt"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose RL adapter vs base model")
    parser.add_argument("--adapter_path", default="outputs/final",
                        help="Path to the RL adapter directory")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of GSM8K questions to evaluate")
    parser.add_argument("--output", default="reports/diagnostic_predictions.json",
                        help="Where to save per-sample predictions")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Additional checkpoint dirs to evaluate (e.g. outputs/step_100)")
    args = parser.parse_args()

    run_diagnostic(args.adapter_path, args.n_samples, args.output, args.checkpoints)
