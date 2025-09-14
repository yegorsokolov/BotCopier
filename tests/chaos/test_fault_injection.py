import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from botcopier.data.loading import _load_logs
from botcopier.scripts import evaluation


def test_network_failure_training(monkeypatch, tmp_path, caplog):
    """Ensure network failures fall back to empty data without crashing."""

    def fail_read_csv(*args, **kwargs):  # simulate network failure
        raise ConnectionError("network down")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)
    caplog.set_level(logging.ERROR)

    df, features, hashes = _load_logs(tmp_path / "trades_raw.csv")
    assert df.empty
    assert features == []
    assert hashes == {}
    assert any("Failed to load logs" in rec.message for rec in caplog.records)


def test_disk_failure_training(monkeypatch, tmp_path, caplog):
    """Ensure disk read errors trigger fallback mode."""

    def fail_read_bytes(self):  # simulate disk failure
        raise OSError("disk error")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    caplog.set_level(logging.ERROR)

    df, features, hashes = _load_logs(tmp_path / "trades_raw.csv")
    assert df.empty
    assert features == []
    assert hashes == {}
    assert any("Failed to load logs" in rec.message for rec in caplog.records)


def test_dependency_failure_inference(monkeypatch, caplog):
    """Metric failures should be logged and return NaN while others work."""

    y = np.array([1, 0])
    p = np.array([0.6, 0.4])

    def bad_metric(y_true, probas, profits):
        raise ImportError("missing dependency")

    def good_metric(y_true, probas, profits):
        return 0.5

    monkeypatch.setattr(
        evaluation,
        "get_metrics",
        lambda selected=None: {"bad": bad_metric, "good": good_metric},
    )
    caplog.set_level(logging.WARNING)

    results = evaluation._classification_metrics(y, p, None)
    assert np.isnan(results["bad"])
    assert results["good"] == 0.5
    assert any("Metric bad failed" in rec.message for rec in caplog.records)


def test_prediction_network_outage(monkeypatch, tmp_path, caplog):
    """Prediction loading should fall back to empty data on network errors."""

    def fail_read_csv(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)
    caplog.set_level(logging.ERROR)

    df = evaluation._load_predictions(tmp_path / "preds.csv")
    assert df.empty
    assert any("Failed to read predictions" in rec.message for rec in caplog.records)


def test_trade_log_network_outage(monkeypatch, tmp_path, caplog):
    """Trade log loading should fall back to empty data on network errors."""

    def fail_read_csv(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)
    caplog.set_level(logging.ERROR)

    df = evaluation._load_actual_trades(tmp_path / "logs.csv")
    assert df.empty
    assert any("Failed to read trade log" in rec.message for rec in caplog.records)
