import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import HAS_NUMPY
from scripts.train_target_clone import train, _load_logs

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is required for training tests")


def _write_log(file: Path):
    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "comment",
        "remaining_lots",
    ]
    rows = [
        [
            "1",
            "2024.01.01 00:00:00",
            "",
            "",
            "OPEN",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "",
            "0.1",
        ],
        [
            "2",
            "2024.01.01 01:00:00",
            "",
            "",
            "OPEN",
            "2",
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.2000",
            "1.1950",
            "1.2100",
            "0",
            "",
            "0.1",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_metrics(file: Path):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["time", "magic", "win_rate", "avg_profit", "trade_count"])
        writer.writerow([
            "2024.01.02 00:00",
            "0",
            "0.5",
            "1.0",
            "2",
        ])


def test_train(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "coefficients" in data
    assert "threshold" in data


def test_train_with_indicators(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, use_sma=True, use_rsi=True, use_macd=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert any(name in data.get("feature_names", []) for name in ["sma", "rsi", "macd"])


def test_train_with_volatility(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    vol_file = tmp_path / "vol.json"
    with open(vol_file, "w") as f:
        json.dump({"2024-01-01": 0.5}, f)

    train(data_dir, out_dir, volatility=json.load(open(vol_file)))

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "volatility" in data.get("feature_names", [])


def test_load_logs_with_metrics(tmp_path: Path):
    data_dir = tmp_path / "logs"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)
    metrics_file = data_dir / "metrics.csv"
    _write_metrics(metrics_file)

    df = _load_logs(data_dir)
    assert "win_rate" in df.columns


def test_train_xgboost(tmp_path: Path):
    pytest.importorskip("xgboost")
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="xgboost", n_estimators=10)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "xgboost"
    assert "coefficients" in data
    assert len(data.get("probability_table", [])) == 24
