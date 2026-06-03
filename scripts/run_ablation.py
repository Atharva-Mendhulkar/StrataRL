# scripts/run_ablation.py
"""
Runs targeting ablation experiments. Each experiment trains for a set number of steps,
evaluates on all 3 benchmarks, and logs the accuracy delta vs baseline.
"""

import argparse, yaml, subprocess, json, os
from pathlib import Path

ABLATIONS = {
    "EXP_01_CONTROL": {
        "description": "Full system — all mechanisms active",
        "overrides": {},
        "hypothesis": "Highest benchmark accuracy. Reference for all deltas.",
    },
    "EXP_02_NO_SAN": {
        "description": "SAN disabled — global normalization only",
        "overrides": {"disable_san": True},
        "hypothesis": "MMLU/StrategyQA accuracy drops. Domain starvation re-emerges.",
    },
    "EXP_03_G4": {
        "description": "G=4 group size (vs G=8 control)",
        "overrides": {"G": 4},
        "hypothesis": "Accuracy drops due to high-variance advantage estimates.",
    },
    "EXP_04_NO_KL": {
        "description": "β=0 unconstrained exploration",
        "overrides": {"beta": 0.0},
        "hypothesis": "Higher peak accuracy but risk of entropy collapse.",
    },
    "EXP_05_NO_CURRICULUM": {
        "description": "Uniform domain sampling (UCB disabled)",
        "overrides": {"disable_curriculum": True},
        "hypothesis": "~2% accuracy drop. Weak domains starve gradient.",
    },
}

def run_ablation(name: str, config: dict, base_config_path: str, steps: int = 50):
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.update(config["overrides"])
    cfg["num_steps"]  = steps
    cfg["run_name"]   = name
    cfg["eval_interval"] = 50 # Lowered for smoke/ablation visibility

    tmp_path = f"/tmp/ablation_{name}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Executing: python m4/m4_train.py --config {tmp_path}")
    
    # Using m4/m4_train.py as it's the local entry point
    result = subprocess.run(
        ["python3", "m4/m4_train.py", "--config", tmp_path],
        env={**os.environ, "PYTHONPATH": ".", "WANDB_MODE": "disabled"},
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="m4/m4_config.yaml")
    parser.add_argument("--steps",       type=int, default=50)
    parser.add_argument("--exp",         default="all",
                        help="Run specific experiment by name, or 'all'")
    args = parser.parse_args()

    to_run = ABLATIONS if args.exp == "all" else {args.exp: ABLATIONS[args.exp]}

    for name, config in to_run.items():
        print(f"\nStarting: {name} — {config['description']}")
        print(f"  Hypothesis: {config['hypothesis']}")
        rc, stdout, stderr = run_ablation(name, config, args.base_config, args.steps)
        print(f"  Exit code: {rc}")
        if rc != 0:
            print(f"  STDERR: {stderr[-500:]}")
        else:
            print(f"  STDOUT summary: {stdout[-500:]}")
