# tests/test_patch_i7.py
import pytest
from training.domain_guard import assert_batch_domain_homogeneity

def test_homogeneous_passes():
    assert_batch_domain_homogeneity(["gsm8k"] * 4)

def test_mixed_raises():
    with pytest.raises(AssertionError, match="DOMAIN HETEROGENEITY"):
        assert_batch_domain_homogeneity(["gsm8k", "mmlu", "gsm8k"])
