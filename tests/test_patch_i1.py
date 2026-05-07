# tests/test_patch_i1.py
import torch, pytest
from rewards.reward_engine import gdpo_normalize, get_gdpo_noise_magnitude, GDPO_CLIP_SPAN

def test_noise_scales_with_clip_range():
    noise = get_gdpo_noise_magnitude(0)
    assert abs(noise - 0.005 * GDPO_CLIP_SPAN) < 1e-6

def test_noise_anneals_monotonically():
    magnitudes = [get_gdpo_noise_magnitude(s) for s in range(0, 201, 10)]
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] >= magnitudes[i+1]

def test_noise_floor_stable_after_200():
    assert abs(get_gdpo_noise_magnitude(200) - get_gdpo_noise_magnitude(500)) < 1e-8

def test_partial_credit_uses_znorm_not_noise():
    partial = torch.tensor([[0.4, 0.6, 0.4, 0.6]])
    z = gdpo_normalize(partial, step=0)
    assert z.abs().max().item() > 0.5  # Z-norm produces wide range

def test_zero_variance_uses_noise():
    same = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    z = gdpo_normalize(same, step=0)
    assert z.abs().max().item() < 0.1  # noise is small magnitude
