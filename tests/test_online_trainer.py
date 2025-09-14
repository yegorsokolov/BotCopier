import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from scripts.online_trainer import OnlineTrainer
from scripts.sequential_drift import PageHinkley


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
    trainer.drift_detector = PageHinkley(delta=0.1, threshold=1.0, min_samples=5)
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


def test_calibration_parameters_evolve(tmp_path: Path):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=2)
    batch1 = [
        {"a": 1.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 1.0, "y": 0},
    ]
    trainer.update(batch1)
    first = json.loads(model_path.read_text()).get("calibration")
    assert first is not None
    batch2 = [
        {"a": 5.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 5.0, "y": 0},
    ]
    trainer.update(batch2)
    second = json.loads(model_path.read_text()).get("calibration")
    assert second is not None
    assert first != second


def test_drift_event_records_hash(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=2)
    batch = [
        {"a": 0.0, "y": 0},
        {"a": 1.0, "y": 1},
    ]
    trainer.update(batch)
    before = json.loads(model_path.read_text())["model_hash"]
    baseline = tmp_path / "baseline.csv"
    recent = tmp_path / "recent.csv"
    baseline.write_text("a\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n")
    recent.write_text("a\n10\n10\n10\n10\n10\n10\n10\n10\n10\n10\n")

    def fake_run(cmd, check):
        data = json.loads(model_path.read_text())
        data["coefficients"] = [42.0]
        model_path.write_text(json.dumps(data))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeThread:
        def __init__(self, target, daemon=True):
            self.target = target

        def start(self):
            try:
                self.target()
            except StopIteration:
                pass

    monkeypatch.setattr(threading, "Thread", FakeThread)

    def fake_sleep(_):
        raise StopIteration

    monkeypatch.setattr(time, "sleep", fake_sleep)

    trainer.start_drift_monitor(
        baseline,
        recent,
        log_dir=tmp_path,
        out_dir=tmp_path,
        files_dir=tmp_path,
        threshold=0.1,
        interval=0.1,
    )

    data = json.loads(model_path.read_text())
    assert data["model_hash"] != before
    assert data["drift_events"] and data["drift_events"][0]["model_hash"] == data["model_hash"]

