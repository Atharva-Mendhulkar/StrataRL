# scripts/audit_config.py
"""
Verifies all patched constants and Kaggle-migration parameters are correctly set.
Run before smoke test. Raises on any misconfiguration.
"""

import yaml, sys, os, argparse

def audit_m4_config(config_path: str):
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        sys.exit(1)
        
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    errors = []

    # Ensure PYTHONPATH includes current dir
    sys.path.append(os.getcwd())

    # I-1: GDPO constants
    try:
        from rewards.reward_engine import GDPO_NOISE_FRACTION, GDPO_ZERO_VAR_THRESH
        if GDPO_NOISE_FRACTION != 0.005:
            errors.append(f"GDPO_NOISE_FRACTION = {GDPO_NOISE_FRACTION}, expected 0.005")
        if GDPO_ZERO_VAR_THRESH != 1e-2:
            errors.append(f"GDPO_ZERO_VAR_THRESH = {GDPO_ZERO_VAR_THRESH}, expected 1e-2")
    except ImportError as e:
        errors.append(f"Could not import GDPO constants: {e}")

    # I-2: No ref_model
    if cfg.get("load_ref_model", False):
        errors.append("load_ref_model=True reintroduces 1.8GB VRAM overhead — must be False")

    # I-3: Recompute interval
    try:
        from training.recompute import RECOMPUTE_INTERVAL
        if not (10 <= RECOMPUTE_INTERVAL <= 50):
            errors.append(f"RECOMPUTE_INTERVAL={RECOMPUTE_INTERVAL} outside [10, 50]")
    except ImportError as e:
        errors.append(f"Could not import RECOMPUTE_INTERVAL: {e}")

    # I-5: SAN and GDPO thresholds must match
    try:
        from training.advantage import SAN_ZERO_VAR_THRESH
        from rewards.reward_engine import GDPO_ZERO_VAR_THRESH
        if SAN_ZERO_VAR_THRESH != GDPO_ZERO_VAR_THRESH:
            errors.append(f"SAN threshold {SAN_ZERO_VAR_THRESH} != GDPO threshold {GDPO_ZERO_VAR_THRESH}")
    except ImportError as e:
        errors.append(f"Could not import SAN or GDPO thresholds: {e}")

    # Kaggle migration parameters check for M4 local run
    if "m4" in config_path and cfg.get("G", 0) > 4:
        errors.append(f"M4 smoke test: G must be <= 4 (got {cfg['G']}). G=8 is for Kaggle only.")

    if errors:
        print("CONFIG AUDIT FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("✓ Config audit passed — all constants correctly patched")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="m4/m4_config.yaml")
    args = parser.parse_args()
    audit_m4_config(args.config)
