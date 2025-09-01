import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import HAS_SB3, HAS_SB3_CONTRIB

if HAS_SB3 and HAS_SB3_CONTRIB:
    from scripts.train_rl_agent import train

pytestmark = pytest.mark.skipif(
    not (HAS_SB3 and HAS_SB3_CONTRIB), reason="sb3-contrib not installed"
)


def _write_log(file: Path) -> None:
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
            "2024.01.01 00:30:00",
            "",
            "",
            "CLOSE",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1050",
            "1.0950",
            "1.1100",
            "5",
            "",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


@pytest.mark.parametrize("algo", ["c51", "qr_dqn"])
def test_train_distributional_agent(tmp_path: Path, algo: str) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades.csv")

    train(
        data_dir,
        out_dir,
        algo=algo,
        training_steps=5,
        learning_rate=0.2,
        gamma=0.95,
    )

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("algo") == algo
    assert "train_metrics" in data
    if algo == "c51":
        assert "value_atoms" in data and "value_distribution" in data
    else:
        assert "value_quantiles" in data
    assert "value_mean" in data and "value_std" in data
