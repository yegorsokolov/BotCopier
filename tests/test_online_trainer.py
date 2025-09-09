import json
import logging
from pathlib import Path

import numpy as np

from scripts.online_trainer import OnlineTrainer


def test_online_trainer_updates(tmp_path: Path):
    model_path = tmp_path / "model.json"

    trainer = OnlineTrainer(model_path=model_path, batch_size=2)
    batch = [
        {"a": 1.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 1.0, "y": 0},
    ]
    trainer.update(batch)

    data = json.loads(model_path.read_text())
    assert set(data["feature_names"]) == {"a", "b"}
    trainer2 = OnlineTrainer(model_path=model_path, batch_size=1)
    before = trainer2.clf.coef_.copy()
    trainer2.update([{"a": 1.0, "b": 1.0, "y": 1}])
    after = trainer2.clf.coef_.copy()
    assert not np.array_equal(before, after)


def test_online_trainer_logs_validation(tmp_path: Path, caplog):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=2)
    batch = [
        {"a": 1.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 1.0, "y": 0},
    ]
    with caplog.at_level(logging.INFO):
        trainer.update(batch)
    events = [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "validation"]
    assert events and "accuracy" in events[0]


def test_conformal_bounds_change(tmp_path: Path):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=2)
    batch = [
        {"a": 1.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 1.0, "y": 0},
    ]
    trainer.update(batch)
    first = json.loads(model_path.read_text())
    lower1, upper1 = first.get("conformal_lower"), first.get("conformal_upper")
    trainer.update([{ "a": 5.0, "b": 5.0, "y": 1 }])
    second = json.loads(model_path.read_text())
    lower2, upper2 = second.get("conformal_lower"), second.get("conformal_upper")
    assert (lower1, upper1) != (lower2, upper2)


def test_regime_shift_triggers_reset(tmp_path: Path, caplog):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=5)
    trainer.cp_window = 4
    trainer.cp_threshold = 1.0
    baseline = [{"a": 0.0, "b": 0.0, "y": 0} for _ in range(5)]
    for _ in range(4):
        trainer.update(baseline)
    old = trainer.clf
    shift = [{"a": 10.0, "b": 10.0, "y": 1} for _ in range(5)]
    with caplog.at_level(logging.INFO):
        trainer.update(shift)
    events = [
        r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "regime_shift"
    ]
    assert events, "regime shift event not logged"
    assert trainer.clf is not old


def test_learning_rate_decay_and_validation(tmp_path: Path):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=2, lr=0.1, lr_decay=0.5)
    batch = [
        {"a": 2.0, "b": 1.0, "y": 1},
        {"a": -1.0, "b": 2.0, "y": 0},
    ]
    X, y = trainer._vectorise(batch)
    try:
        acc0 = float(np.mean(trainer.clf.predict(X) == y))
    except Exception:
        acc0 = 0.0
    trainer.update(batch)
    X, y = trainer._vectorise(batch)
    acc1 = float(np.mean(trainer.clf.predict(X) == y))
    trainer.update(batch)
    X, y = trainer._vectorise(batch)
    acc2 = float(np.mean(trainer.clf.predict(X) == y))
    assert trainer.lr_history[0] > trainer.lr_history[1]
    assert acc2 >= acc1 >= acc0

