# scripts/generate_kaggle_config.py
"""
Takes M4 validated config and produces Kaggle-ready config.
Never hand-edit — always generate from M4 config to preserve inheritance.
"""

import yaml
from pathlib import Path

def generate_kaggle_config(
    m4_config_path:     str = "m4/m4_config.yaml",
    kaggle_config_path: str = "configs/exp_01_kaggle.yaml",
):
    with open(m4_config_path) as f:
        base = yaml.safe_load(f)

    kaggle = {
        # ── Model ──────────────────────────────────────────────────────────
        "model_id":       "Qwen/Qwen2.5-3B-Instruct",
        "device":         "cuda",
        "dtype":          "bfloat16",

        # ── Training ────────────────────────────────────────────────────────
        "num_steps":      1000,
        "G":              8,
        "batch_size":     4,
        "grad_accum":     4,

        # ── GRPO ─────────────────────────────────────────────────────────────
        "beta":           0.010,
        "clip_eps":       0.20,

        # ── LoRA (Unsloth, Kaggle only) ─────────────────────────────────────
        "lora_r":         32,
        "lora_alpha":     64,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "load_in_4bit":   True,

        # ── I-2: No reference model ─────────────────────────────────
        "load_ref_model": False,

        # ── Rollout ──────────────────────────────────────────────────────────
        "max_new_tokens": 2048,
        "min_new_tokens": 100,
        "temperature":    0.85,
        "top_p":          0.95,
        "vllm_gpu_util":  0.50,

        # ── Rewards ──────────────────────────────────────────────────────────
        "w_outcome":      0.70,
        "w_struct":       0.30,
        "reward_clip":    2.0,

        # ── Curriculum ───────────────────────────────────────────────────────
        "domains":        ["gsm8k", "mmlu", "strategyqa"],
        "ucb_c":          0.5,

        # ── Eval ─────────────────────────────────────────────────────────────
        "eval_interval":  100,

        # ── Monitoring ───────────────────────────────────────────────────────
        "wandb_project":  "stratarl_kaggle_3b",
    }

    Path(kaggle_config_path).parent.mkdir(exist_ok=True)
    with open(kaggle_config_path, "w") as f:
        yaml.dump(kaggle, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Kaggle config generated: {kaggle_config_path}")
    return kaggle_config_path

if __name__ == "__main__":
    generate_kaggle_config()
