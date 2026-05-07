# tests/test_patch_i3.py
from training.recompute import should_recompute, RECOMPUTE_INTERVAL

def test_fires_at_correct_intervals():
    fired = [s for s in range(200) if should_recompute(s)]
    assert fired == list(range(0, 200, RECOMPUTE_INTERVAL))

def test_overhead_under_5pct():
    fired = sum(1 for s in range(100) if should_recompute(s))
    assert fired / 100 <= 0.05

def test_interval_in_reasonable_range():
    assert 10 <= RECOMPUTE_INTERVAL <= 50
