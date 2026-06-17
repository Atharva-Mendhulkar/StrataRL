# eval/benchmark_eval.py

import torch
import numpy as np
import re
from datasets import load_dataset
from typing import Dict, List, Tuple
from sympy import sympify, simplify

# ── Answer extractors ─────────────────────────────────────────────────────────

THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE  = re.compile(r"\\boxed\{([^}]+)\}")

def extract_answer(completion: str) -> str | None:
    m = ANSWER_RE.search(completion)
    if m: return m.group(1).strip()
    m = BOXED_RE.search(completion)
    if m: return m.group(1).strip()
    return None

def extract_think_length(completion: str) -> int:
    m = THINK_RE.search(completion)
    if not m: return 0
    return len(m.group(1).split())  # word count

# ── Verifiers ─────────────────────────────────────────────────────────────────

def verify_math(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    # Remove any non-numeric stuff if it's a simple number
    p = predicted.strip().replace("$", "").replace(",", "")
    g = ground_truth.strip().replace("$", "").replace(",", "")
    
    try:
        if abs(float(p) - float(g)) < 1e-6: return True
    except (ValueError, TypeError): pass
    
    try:
        pred_expr  = sympify(p, evaluate=True)
        truth_expr = sympify(g, evaluate=True)
        if simplify(pred_expr - truth_expr) == 0: return True
    except Exception: pass
    
    return p.lower() == g.lower()

def verify_mcq(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    # Extract first letter A-D
    m = re.search(r"[A-D]", predicted.upper())
    if m:
        return m.group(0) == ground_truth.strip().upper()[:1]
    return predicted.strip().upper()[:1] == ground_truth.strip().upper()[:1]

def verify_bool(predicted: str, ground_truth: str) -> bool:
    if not predicted: return False
    p = predicted.strip().lower()
    g = ground_truth.strip().lower()
    p_bool = p in ("yes", "true", "1")
    g_bool = g in ("yes", "true", "1")
    return p_bool == g_bool


# ── Benchmark configs ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "gsm8k": {
        "hf_path":    "gsm8k",
        "hf_name":    "main",
        "split":      "test",
        "n_samples":  500,
        "verifier":   verify_math,
        "answer_key": "answer",
        "baseline_lit": 0.867, # Literature baseline for 3B
    },
    "mmlu": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "all",
        "split":      "test",
        "n_samples":  500,
        "verifier":   verify_mcq,
        "answer_key": "answer",
        "baseline_lit": 0.644,
    },
    "strategyqa": {
        "hf_path":    "tasksource/strategy-qa",
        "hf_name":    None,
        "split":      "train", # tasksource only has train
        "n_samples":  490,
        "verifier":   verify_bool,
        "answer_key": "answer",
        "baseline_lit": 0.650,
    },
}

MMLU_INT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

def format_gsm8k_prompt(item: dict) -> Tuple[str, str]:
    q  = item["question"]
    gt = item["answer"].split("####")[-1].strip()
    prompt = (
        "You are a precise math reasoning assistant. "
        "Solve the following problem step by step. "
        "Show your internal reasoning process inside <think> tags. "
        "Place only your final numeric answer inside <answer> tags.\n\n"
        f"Question: {q}\n"
    )
    return prompt, gt

def format_mmlu_prompt(item: dict) -> Tuple[str, str]:
    q       = item["question"]
    choices = item["choices"]
    gt_int  = item["answer"]
    gt      = MMLU_INT_TO_LETTER[gt_int]
    options = "\n".join(f"{MMLU_INT_TO_LETTER[i]}. {c}" for i, c in enumerate(choices))
    prompt  = (
        "You are a knowledgeable assistant. "
        "Answer the following multiple choice question. "
        "Think step by step inside <think> tags. "
        "Then, write only the single letter (A, B, C, or D) corresponding to the correct answer inside <answer> tags.\n\n"
        f"Question: {q}\n{options}\n"
    )
    return prompt, gt

def format_strategyqa_prompt(item: dict) -> Tuple[str, str]:
    q  = item["question"]
    gt = "yes" if item["answer"] else "no"
    prompt = (
        "You are a logical reasoning assistant. "
        "Answer the following yes/no question. "
        "Explain your reasoning inside <think> tags. "
        "Finally, write only 'yes' or 'no' inside <answer> tags.\n\n"
        f"Question: {q}\n"
    )
    return prompt, gt

FORMATTERS = {
    "gsm8k":       format_gsm8k_prompt,
    "mmlu":        format_mmlu_prompt,
    "strategyqa":  format_strategyqa_prompt,
}


# ── Evaluation runner ─────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    def __init__(self, generate_fn):
        self.generate = generate_fn

    def run_benchmark(
        self,
        bench_name: str,
        greedy_only: bool = False,
        n_samples: int = None,
    ) -> Dict:
        cfg       = BENCHMARKS[bench_name]
        formatter = FORMATTERS[bench_name]

        # Use trust_remote_code=True for CAIS/MMLU if needed, but tasksource is safe
        try:
            ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
        except Exception:
            ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"], trust_remote_code=True)
            
        actual_n = n_samples if n_samples is not None else cfg["n_samples"]
        
        # If strategyqa on tasksource, take from the end to avoid train overlap
        if bench_name == "strategyqa" and cfg["hf_path"] == "tasksource/strategy-qa":
            ds = ds.select(range(len(ds) - actual_n, len(ds)))
        else:
            ds = ds.select(range(min(actual_n, len(ds))))

        greedy_correct = 0
        think_lengths  = []
        total          = len(ds)

        for idx, item in enumerate(ds):
            prompt, gt = formatter(item)
            greedy_completions = self.generate([prompt], temperature=0.0, max_tokens=256)
            comp = greedy_completions[0]
            pred = extract_answer(comp)
            correct = cfg["verifier"](pred, gt)
            if correct:
                greedy_correct += 1
            think_lengths.append(extract_think_length(comp))
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"    [{bench_name}] {idx+1}/{total}  running_acc={greedy_correct/(idx+1):.3f}", flush=True)

        greedy_acc  = greedy_correct / total
        avg_think   = np.mean(think_lengths) if think_lengths else 0

        return {
            "benchmark":     bench_name,
            "n_samples":     total,
            "greedy_acc":    greedy_acc,
            "avg_think_len": avg_think,
            "baseline_lit":  cfg["baseline_lit"],
        }

    def run_all(self, step: int, greedy_only: bool = False, n_samples: int = None) -> Dict:
        results = {}
        for bench in BENCHMARKS:
            r = self.run_benchmark(bench, greedy_only=greedy_only, n_samples=n_samples)
            results[bench] = r
            print(
                f"[Step {step}] {bench}: {r['greedy_acc']:.4f} "
                f"(lit_baseline: {r['baseline_lit']:.4f}, avg_think: {r['avg_think_len']:.1f} words)"
            )
        return results
