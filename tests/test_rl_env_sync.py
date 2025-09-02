import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import HAS_SB3

if HAS_SB3:
    from scripts.train_rl_agent import train

pytestmark = pytest.mark.skipif(not HAS_SB3, reason="stable-baselines3 not installed")


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


def test_parallel_env_rewards(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades_0.csv")

    train(
        data_dir,
        out_dir,
        algo="dqn",
        training_steps=5,
        num_envs=2,
    )

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    rewards = data.get("episode_rewards")
    assert isinstance(rewards, list)
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(rewards[1])
    assert data.get("avg_reward") == pytest.approx(sum(rewards) / 2)
