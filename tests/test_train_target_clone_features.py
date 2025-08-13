import csv
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_target_clone import _extract_features, train


def _write_sample_log(file: Path):
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
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
        "open_time",
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "sl_hit_dist",
        "tp_hit_dist",
        "commission",
        "swap",
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
            "2",
            "",
            "0.1",
            "0.0001",
            "100",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
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
            "3",
            "",
            "0.1",
            "0.0002",
            "200",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_feature_extraction_basic():
    rows = [
        {
            "event_time": datetime(2024, 1, 1, 0, 0),
            "action": "OPEN",
            "symbol": "EURUSD",
            "order_type": "0",
            "lots": 0.1,
            "price": 1.1000,
            "sl": 1.0950,
            "tp": 1.1100,
            "profit": 0,
            "spread": 2,
            "slippage": 0.0001,
            "equity": 1000,
            "margin_level": 150,
        }
    ]
    feats, *_ = _extract_features(rows)
    assert "hour_sin" in feats[0] and "hour_cos" in feats[0]
    assert "dow_sin" in feats[0] and "dow_cos" in feats[0]
    assert "spread" in feats[0]
    assert "slippage" in feats[0]
    assert "equity" in feats[0] and "margin_level" in feats[0]
    assert "sl_dist" in feats[0] and "tp_dist" in feats[0]
    assert "sl_hit_dist" in feats[0] and "tp_hit_dist" in feats[0]


def test_model_serialization(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sample_log(data_dir / "trades_sample.csv")

    train(data_dir, out_dir)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "feature_names" in data
    assert "mean" in data
    assert "std" in data
    assert "threshold" in data
    assert "val_accuracy" in data
    assert "spread" in data.get("feature_names", [])
    assert "slippage" in data.get("feature_names", [])


def test_perf_budget_disables_heavy_features(caplog):
    rows = [
        {
            "event_time": datetime(2024, 1, 1, 0, 0),
            "action": "OPEN",
            "symbol": "EURUSD",
            "order_type": "0",
            "lots": 0.1,
            "price": 1.1000,
            "sl": 1.0950,
            "tp": 1.1100,
            "profit": 0,
            "spread": 2,
        },
        {
            "event_time": datetime(2024, 1, 1, 0, 1),
            "action": "OPEN",
            "symbol": "EURUSD",
            "order_type": "0",
            "lots": 0.1,
            "price": 1.1002,
            "sl": 1.0950,
            "tp": 1.1100,
            "profit": 0,
            "spread": 2,
        },
    ]
    with caplog.at_level(logging.INFO):
        feats, *_ = _extract_features(
            rows,
            use_atr=True,
            use_bollinger=True,
            use_stochastic=True,
            use_adx=True,
            higher_timeframes=["M5"],
            perf_budget=1e-9,
        )
    enabled_msgs = [r.message for r in caplog.records if "Enabled features" in r.message]
    assert enabled_msgs and "atr" not in enabled_msgs[-1]
    assert "atr" not in feats[-1]
