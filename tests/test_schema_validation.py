import csv
from pathlib import Path

import pytest

from botcopier.exceptions import DataError
from botcopier.scripts.evaluation import evaluate


def test_evaluate_missing_prediction_field(tmp_path: Path) -> None:
    pred = tmp_path / "preds.csv"
    with open(pred, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots"])
        writer.writerow(["2024.01.01 00:00:00", "EURUSD", "buy", "0.1"])

    log = tmp_path / "trades.csv"
    with open(log, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["event_time", "action", "ticket", "symbol", "order_type", "lots", "profit"]
        )
        writer.writerow(["2024.01.01 00:00:05", "OPEN", "1", "EURUSD", "0", "0.1", "0"])
        writer.writerow(
            ["2024.01.01 00:01:00", "CLOSE", "1", "EURUSD", "0", "0.1", "1"]
        )

    with pytest.raises(DataError):
        evaluate(pred, log, window=60)
