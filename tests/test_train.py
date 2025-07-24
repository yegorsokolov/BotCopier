import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_target_clone import train


def _write_log(file: Path):
    fields = [
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
    ]
    rows = [
        [
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
        ],
        [
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
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


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
