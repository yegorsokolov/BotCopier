import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.active_validator import ActiveValidator


def test_retrain_on_metric_decay():
    calls: list[str] = []

    validator = ActiveValidator(threshold=0.8, retrain_cb=lambda: calls.append("retrained"), window=2)
    truth = [1, 0, 1, 1]
    good = [1, 0, 1, 1]
    bad = [0, 0, 1, 1]  # accuracy 0.75

    validator.evaluate(good, truth)
    validator.evaluate(bad, truth)  # rolling avg 0.875 -> no retrain
    assert calls == []
    validator.evaluate(bad, truth)  # rolling avg 0.75 -> retrain
    assert calls == ["retrained"]


def test_no_retrain_when_metrics_good():
    calls: list[str] = []

    validator = ActiveValidator(threshold=0.5, retrain_cb=lambda: calls.append("r"))
    truth = [1, 0]
    preds = [1, 1]  # accuracy 0.5 which equals threshold
    metric = validator.evaluate(preds, truth)
    assert metric == 0.5
    assert calls == []


def test_demote_after_persistent_failures():
    calls: list[str] = []

    validator = ActiveValidator(
        threshold=0.9,
        retrain_cb=lambda: calls.append("retrain"),
        demote_cb=lambda: calls.append("demote"),
        window=3,
        patience=2,
    )
    truth = [1, 1, 1]
    bad = [0, 0, 0]
    validator.evaluate(bad, truth)
    validator.evaluate(bad, truth)
    assert calls == ["retrain", "retrain", "demote"]
