import csv
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rl_agent import train


def _write_synthetic(file: Path) -> None:
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
    rows = []
    ticket = 1
    # majority trades: hour=0, action=0, profit=1
    for _ in range(18):
        open_row = [
            str(ticket * 2 - 1),
            "2024.01.01 00:00:00",
            "",
            "",
            "OPEN",
            str(ticket),
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
        ]
        close_row = [
            str(ticket * 2),
            "2024.01.01 00:10:00",
            "",
            "",
            "CLOSE",
            str(ticket),
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1010",
            "1.0950",
            "1.1100",
            "1",
            "",
            "0",
        ]
        rows.extend([open_row, close_row])
        ticket += 1
    # minority trades: hour=1, action=1, profit=10
    for _ in range(2):
        open_row = [
            str(ticket * 2 - 1),
            "2024.01.01 01:00:00",
            "",
            "",
            "OPEN",
            str(ticket),
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "",
            "0.1",
        ]
        close_row = [
            str(ticket * 2),
            "2024.01.01 01:10:00",
            "",
            "",
            "CLOSE",
            str(ticket),
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.1010",
            "1.0950",
            "1.1100",
            "10",
            "",
            "0",
        ]
        rows.extend([open_row, close_row])
        ticket += 1
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_prioritized_sampling_converges_faster(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_pri = tmp_path / "pri"
    out_uni = tmp_path / "uni"
    data_dir.mkdir()
    out_pri.mkdir()
    out_uni.mkdir()
    _write_synthetic(data_dir / "trades.csv")

    np.random.seed(0)
    train(
        data_dir,
        out_pri,
        algo="qlearn",
        training_steps=5,
        epsilon=0.0,
    )
    np.random.seed(0)
    train(
        data_dir,
        out_uni,
        algo="qlearn",
        training_steps=5,
        epsilon=0.0,
        replay_alpha=0.0,
        replay_beta=0.0,
    )

    with open(out_pri / "model.json") as f:
        pri = json.load(f)
    with open(out_uni / "model.json") as f:
        uni = json.load(f)
    assert pri["train_accuracy"] >= uni["train_accuracy"]
