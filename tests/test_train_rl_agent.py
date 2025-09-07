import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import HAS_SB3, HAS_SB3_CONTRIB

if HAS_SB3:
    from scripts.train_rl_agent import train
    from scripts.train_rl_agent import _build_dataset
    import stable_baselines3 as sb3
    import torch

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

@pytest.mark.parametrize("algo", ["dqn", "a2c"])
def test_train_rl_agent_sb3(tmp_path: Path, algo: str) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades_0.csv")

    start_model = out_dir / "start.json"
    start = {
        "model_id": "sup_model",
        "coefficients": [0.1, 0.1],
        "intercept": 0.0,
        "feature_names": ["hour_sin", "hour_cos"],
    }
    with open(start_model, "w") as f:
        json.dump(start, f)

    train(
        data_dir,
        out_dir,
        start_model=start_model,
        algo=algo,
        training_steps=10,
        learning_rate=0.2,
        gamma=0.95,
        num_envs=2,
    )

    model_file = out_dir / "model.json"
    weights_file = out_dir / "model_weights.zip"
    assert model_file.exists()
    assert weights_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("algo") == algo
    assert data.get("training_steps") == 10
    assert data.get("learning_rate") == pytest.approx(0.2)
    assert data.get("gamma") == pytest.approx(0.95)
    assert "avg_reward" in data
    if algo == "dqn":
        assert data.get("init_model_id") == "sup_model"


@pytest.mark.skipif(
    not (HAS_SB3 and HAS_SB3_CONTRIB),
    reason="stable-baselines3 or sb3-contrib not installed",
)
def test_intrinsic_reward_increases_entropy(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    base_out = tmp_path / "base"
    intr_out = tmp_path / "intr"
    for p in [data_dir, base_out, intr_out]:
        p.mkdir()
    _write_log(data_dir / "trades_0.csv")

    start_model = base_out / "start.json"
    start = {
        "model_id": "sup_model",
        "coefficients": [0.1, 0.1],
        "intercept": 0.0,
        "feature_names": ["hour_sin", "hour_cos"],
    }
    with open(start_model, "w") as f:
        json.dump(start, f)

    train(
        data_dir,
        base_out,
        start_model=start_model,
        algo="a2c",
        training_steps=20,
        learning_rate=0.2,
        gamma=0.95,
    )

    train(
        data_dir,
        intr_out,
        start_model=start_model,
        algo="a2c",
        training_steps=20,
        learning_rate=0.2,
        gamma=0.95,
        intrinsic_reward=True,
        intrinsic_reward_weight=1.0,
    )

    states, _, _, _, _ = _build_dataset(data_dir, None, None)
    obs = torch.tensor(states, dtype=torch.float32)
    base_model = sb3.A2C.load(str(base_out / "model_weights.zip"))
    intr_model = sb3.A2C.load(str(intr_out / "model_weights.zip"))
    with torch.no_grad():
        base_ent = base_model.policy.get_distribution(obs).entropy().mean().item()
        intr_ent = intr_model.policy.get_distribution(obs).entropy().mean().item()
    assert intr_ent >= base_ent
