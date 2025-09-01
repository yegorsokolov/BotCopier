import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation import evaluate


def _write_prediction(file: Path, ts: str, direction: str = "buy"):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots"])
        writer.writerow([ts, "EURUSD", direction, "0.1"])


def _write_actual(file: Path, ts_open: str, ts_close: str, order_type: str = "0"):
    fields = [
        "event_time",
        "action",
        "ticket",
        "symbol",
        "order_type",
        "lots",
        "profit",
    ]
    rows = [
        [ts_open, "OPEN", "1", "EURUSD", order_type, "0.1", "0"],
        [ts_close, "CLOSE", "1", "EURUSD", order_type, "0.1", "10"],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_evaluate(tmp_path: Path):
    pred_file = tmp_path / "preds.csv"
    log_file = tmp_path / "trades.csv"
    _write_prediction(pred_file, "2024.01.01 00:00:00")
    _write_actual(log_file, "2024.01.01 00:00:05", "2024.01.01 00:01:00")

    stats = evaluate(pred_file, log_file, window=60)

    assert stats["matched_events"] == 1
    assert stats["predicted_events"] == 1
    assert stats["accuracy"] == 1.0
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0
    assert stats["profit_factor"] == float("inf")
    assert stats["sharpe_ratio"] == 0.0
    assert stats["sortino_ratio"] == 0.0
    assert stats["expectancy"] == 10
    assert stats["expected_return"] == 10
    assert stats["downside_risk"] == 0
    assert stats["risk_reward"] == 10


def test_direction_mapping(tmp_path: Path):
    cases = [
        ("1", "0"),
        ("buy", "0"),
        ("0", "1"),
        ("-1", "1"),
        ("sell", "1"),
        ("", "0"),
    ]

    for direction_value, order_type in cases:
        pred_file = tmp_path / f"pred_{direction_value or 'empty'}.csv"
        log_file = tmp_path / f"trades_{direction_value or 'empty'}.csv"
        _write_prediction(pred_file, "2024.01.01 00:00:00", direction_value)
        _write_actual(
            log_file,
            "2024.01.01 00:00:05",
            "2024.01.01 00:01:00",
            order_type,
        )

        stats = evaluate(pred_file, log_file, window=60)
        assert stats["matched_events"] == 1
