from pathlib import Path
import csv
import pytest
import sys
import types

sys.modules["pyarrow"] = types.SimpleNamespace(__version__="0")

# Skip test if training module cannot be imported due to missing dependencies
train_mod = pytest.importorskip("scripts.train_target_clone")
_load_logs = train_mod._load_logs


def _write_log(path: Path):
    fields = [
        "schema_version",
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
    ]
    good = [
        1,
        1,
        "2024-01-01 00:00:00",
        "",
        "",
        "OPEN",
        1,
        0,
        "src",
        "EURUSD",
        0,
        0.1,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "",
        0.0,
        0.0,
        0,
        "",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    bad = good.copy()
    bad[1] = "bad"  # invalid event_id
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerow(good)
        writer.writerow(bad)


def test_load_logs_validates_rows(tmp_path: Path):
    _write_log(tmp_path / "trades_1.csv")
    df, commits, checksums = _load_logs(tmp_path)
    assert len(df) == 1
    invalid_file = tmp_path / "invalid_rows.csv"
    assert invalid_file.exists()
    with invalid_file.open() as f:
        reader = list(csv.reader(f))
    assert len(reader) == 2  # header + one invalid row
