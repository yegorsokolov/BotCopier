import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_predictions import evaluate


def _write_prediction(file: Path, ts: str):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots"])
        writer.writerow([ts, "EURUSD", "buy", "0.1"])


def _write_actual(file: Path, ts_open: str, ts_close: str):
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
        [ts_open, "OPEN", "1", "EURUSD", "0", "0.1", "0"],
        [ts_close, "CLOSE", "1", "EURUSD", "0", "0.1", "10"],
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
