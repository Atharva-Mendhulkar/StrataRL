# eval/latency_profiler.py

import time
import torch
import numpy as np
from typing import Dict, List

def profile_inference_latency(
    generate_fn,
    prompts: List[str],
    temperatures: List[float] = [0.0, 0.85],
    max_tokens_list: List[int] = [512, 2048],
    n_warmup: int = 3,
    n_measure: int = 20,
) -> Dict:
    """
    Measures:
    - Tokens per second at different generation settings
    - End-to-end latency per prompt
    - VRAM usage during inference
    """
    results = {}

    for temp in temperatures:
        for max_tok in max_tokens_list:
            key = f"temp{temp}_maxtok{max_tok}"

            # Warmup
            for _ in range(n_warmup):
                generate_fn(prompts[:1], temperature=temp, max_tokens=max_tok)

            # Measure
            latencies = []
            token_counts = []

            for _ in range(n_measure):
                prompt = prompts[_ % len(prompts)]
                if torch.cuda.is_available(): torch.cuda.synchronize()
                elif torch.backends.mps.is_available(): torch.mps.synchronize()

                start = time.perf_counter()
                completions = generate_fn([prompt], temperature=temp, max_tokens=max_tok)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                elif torch.backends.mps.is_available(): torch.mps.synchronize()
                end   = time.perf_counter()

                latency = end - start
                n_tokens = len(completions[0].split())  # word count proxy
                latencies.append(latency)
                token_counts.append(n_tokens)

            results[key] = {
                "mean_latency_sec":   np.mean(latencies),
                "p95_latency_sec":    np.percentile(latencies, 95),
                "mean_tokens":        np.mean(token_counts),
                "tokens_per_sec":     np.mean(token_counts) / np.mean(latencies),
                "temperature":        temp,
                "max_tokens":         max_tok,
            }

    # VRAM measurement
    if torch.cuda.is_available():
        results["vram_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
    elif torch.backends.mps.is_available():
        results["vram_gb"] = torch.mps.current_allocated_memory() / (1024 ** 3)

    return results


def compare_baseline_vs_rl(
    baseline_gen_fn,
    rl_gen_fn,
    prompts: List[str],
) -> Dict:
    """Compare latency and token efficiency: baseline SFT vs RL-trained model."""
    baseline_profile = profile_inference_latency(baseline_gen_fn, prompts)
    rl_profile       = profile_inference_latency(rl_gen_fn, prompts)

    comparison = {
        "baseline": baseline_profile,
        "rl":       rl_profile,
    }

    # Key ratio
    for key in baseline_profile:
        if isinstance(baseline_profile[key], dict):
            b_tps = baseline_profile[key].get("tokens_per_sec", 1)
            r_tps = rl_profile[key].get("tokens_per_sec", 1)
            comparison[f"overhead_ratio_{key}"] = r_tps / b_tps

    return comparison
