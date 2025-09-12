import csv
from pathlib import Path

from botcopier.scripts.evaluation import evaluate


def _write_prediction(file: Path, rows):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots", "probability"])
        writer.writerows(rows)


def _write_actual(file: Path, rows):
    fields = ["event_time", "action", "ticket", "symbol", "order_type", "lots", "profit"]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_fee_adjusted_metrics(tmp_path: Path):
    pred_file = tmp_path / "preds.csv"
    log_file = tmp_path / "trades.csv"
    preds = [
        ["2024.01.01 00:00:00", "EURUSD", "buy", "0.1", 0.9],
        ["2024.01.02 00:00:00", "EURUSD", "buy", "0.1", 0.9],
    ]
    _write_prediction(pred_file, preds)
    trades = [
        ["2024.01.01 00:00:05", "OPEN", "1", "EURUSD", "0", "0.1", "0"],
        ["2024.01.01 00:01:00", "CLOSE", "1", "EURUSD", "0", "0.1", "10"],
        ["2024.01.02 00:00:05", "OPEN", "2", "EURUSD", "0", "0.1", "0"],
        ["2024.01.02 00:01:00", "CLOSE", "2", "EURUSD", "0", "0.1", "-5"],
    ]
    _write_actual(log_file, trades)

    stats = evaluate(pred_file, log_file, window=60, fee_per_trade=5.0)
    assert stats["sharpe_ratio_net"] < stats["sharpe_ratio"]
    assert stats["sortino_ratio_net"] < stats["sortino_ratio"]
