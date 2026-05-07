# tests/test_patch_i4.py
from monitoring.monitor import DeltaOSTracker, DIVERSE_NONSENSE_TRIGGER, DELTA_OS_ALERT_THRESHOLD

def test_no_alert_healthy_training():
    t = DeltaOSTracker()
    for s in range(60):
        r = t.update(0.7, 0.6, s)
    assert r["delta_os/status"] == "OK"

def test_attack_detected_after_trigger():
    t = DeltaOSTracker()
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        r = t.update(0.001, 0.8, s)
    assert r["delta_os/status"] == "ATTACK"

def test_no_false_alarm_before_trigger():
    t = DeltaOSTracker()
    results = [t.update(0.001, 0.8, s) for s in range(DIVERSE_NONSENSE_TRIGGER - 1)]
    assert results[-1]["delta_os/status"] != "ATTACK"

def test_recovery_clears_attack():
    t = DeltaOSTracker()
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        t.update(0.001, 0.8, s)
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5, DIVERSE_NONSENSE_TRIGGER + 60):
        r = t.update(0.7, 0.8, s)
    assert r["delta_os/status"] != "ATTACK"

def test_weight_override_active_during_attack():
    t = DeltaOSTracker()
    assert t.get_outcome_weight_override() is None
    for s in range(DIVERSE_NONSENSE_TRIGGER + 5):
        t.update(0.001, 0.8, s)
    assert t.get_outcome_weight_override() == 0.85
