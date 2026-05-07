# tests/test_patch_i5.py
import torch, pytest
from training.advantage import compute_san_advantages, SAN_ZERO_VAR_THRESH
from rewards.reward_engine import GDPO_ZERO_VAR_THRESH

def test_zero_variance_centers_without_scaling():
    r = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.abs().max().item() < 1e-5

def test_partial_credit_is_dampened():
    r = torch.tensor([[0.49, 0.51, 0.49, 0.51]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.std().item() < 0.8

def test_high_variance_full_normalization():
    r = torch.tensor([[0.0, 1.0, 0.0, 1.0]] * 2)
    a = compute_san_advantages(r, ["gsm8k", "gsm8k"])
    assert a.std().item() > 0.5

def test_threshold_matches_gdpo():
    assert SAN_ZERO_VAR_THRESH == GDPO_ZERO_VAR_THRESH, (
        f"Mismatch: SAN={SAN_ZERO_VAR_THRESH}, GDPO={GDPO_ZERO_VAR_THRESH}"
    )

def test_cross_stratum_bias_prevented():
    r = torch.tensor([
        [0.9, 0.9, 0.8, 0.9], [0.9, 0.9, 0.8, 0.9],
        [0.4, 0.3, 0.4, 0.5], [0.4, 0.3, 0.4, 0.5],
    ])
    d = ["gsm8k", "gsm8k", "strategyqa", "strategyqa"]
    a = compute_san_advantages(r, d)
    assert abs(a[2:4].mean().item()) < 0.3
