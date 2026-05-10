# scripts/generate_kaggle_config.py
"""
Generates the production Kaggle config (configs/exp_01_kaggle.yaml).
Never hand-edit the output — always regenerate from this script.
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
        # Model
        "model_id":     "Qwen/Qwen2.5-3B-Instruct",
        "device":       "cuda",
        "dtype":        "float16",

        # Training
        "num_steps":    200,
        "G":            4,
        "batch_size":   2,
        "grad_accum":   4,
        "lr":           5e-6,
        "max_new_tokens": 512,
        "min_new_tokens": 32,

        # Phase-dependent GRPO (calibration → production)
        # steps 0-100: tight trust region
        "beta_phase1":       0.015,
        "clip_eps_phase1":   0.15,
        # steps 101+: relaxed
        "beta_phase2":       0.010,
        "clip_eps_phase2":   0.20,
        "beta_switch_step":  100,
        # Convenience fields (training/train.py reads these at step 0)
        "beta":         0.015,
        "clip_eps":     0.15,

        # LoRA (full projection set for 3B)
        "lora_r":          32,
        "lora_alpha":      64,
        "target_modules":  [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],

        # I-2: No reference model — π_ref = π_old (saves 1.8 GB VRAM)
        "load_ref_model": False,
        "load_in_4bit":   True,

        # Rollout
        "temperature":         0.85,
        "top_p":               0.95,
        "vllm_gpu_util":       0.50,
        "vllm_sync_interval":  10,

        # I-3: Recompute interval (25 steps)
        "recompute_interval":  25,

        # Rewards
        "w_outcome":    0.70,
        "w_struct":     0.30,
        "reward_clip":  2.0,

        # Curriculum
        "domains":             ["gsm8k", "mmlu", "strategyqa"],
        "samples_per_domain":  100,
        "ucb_c":               0.5,

        # Eval & monitoring
        "eval_interval":   100000,
        "wandb_project":   "stratarl_kaggle_3b",
    }

    Path(kaggle_config_path).parent.mkdir(exist_ok=True)
    with open(kaggle_config_path, "w") as f:
        yaml.dump(kaggle, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Kaggle config generated: {kaggle_config_path}")
    return kaggle_config_path


if __name__ == "__main__":
    generate_kaggle_config()
