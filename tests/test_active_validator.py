import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.active_validator import ActiveValidator


def test_retrain_on_metric_decay():
    calls: list[str] = []

    def retrain():
        calls.append("retrained")

    validator = ActiveValidator(threshold=0.8, retrain_cb=retrain)
    truth = [1, 0, 1, 1]
    good = [1, 0, 1, 1]
    bad = [0, 0, 1, 1]  # accuracy 0.75

    m1 = validator.evaluate(good, truth)
    assert m1 == 1.0
    m2 = validator.evaluate(bad, truth)
    assert m2 == 0.75
    assert calls == ["retrained"]
    m3 = validator.evaluate(good, truth)
    assert m3 == 1.0
    assert calls == ["retrained"]


def test_no_retrain_when_metrics_good():
    calls: list[str] = []

    validator = ActiveValidator(threshold=0.5, retrain_cb=lambda: calls.append("r"))
    truth = [1, 0]
    preds = [1, 1]  # accuracy 0.5 which equals threshold
    metric = validator.evaluate(preds, truth)
    assert metric == 0.5
    assert calls == []
