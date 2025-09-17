import asyncio
import json

import numpy as np
import pandas as pd

from botcopier.scripts.online_trainer import OnlineTrainer
from scripts.sequential_drift import PageHinkley


class DummyController:
    def __init__(self) -> None:
        self.sampled = False

    def update(self, action, reward, alpha: float = 0.1) -> None:  # pragma: no cover - simple stub
        return None

    def sample_action(self):  # pragma: no cover - deterministic sample
        self.sampled = True
        return (("feat",), "confidence_weighted"), []


async def _tick_stream():
    for _ in range(16):
        yield {"feat": 0.0, "y": 0}
    for _ in range(16):
        yield {"feat": 1.0, "y": 1}
    for i in range(16):
        yield {"feat": 1.0, "y": i % 2}
    for _ in range(16):
        yield {"feat": 0.0, "y": 0}


def test_tick_stream_buffer_and_adaptive_refit(tmp_path):
    ring_path = tmp_path / "ticks.ring"
    model_path = tmp_path / "model.json"
    buffer_path = tmp_path / "data/live_ticks.parquet"
    controller = DummyController()
    trainer = OnlineTrainer(
        model_path=model_path,
        batch_size=16,
        controller=controller,
        tick_buffer_path=buffer_path,
    )
    asyncio.run(trainer.consume_ticks(_tick_stream(), ring_path))
    assert buffer_path.exists()
    df = pd.read_parquet(buffer_path)
    assert len(df) >= 64
    assert controller.sampled
    assert trainer.model_type == "confidence_weighted"


async def _drift_stream():
    for _ in range(6):
        yield {"feat": -0.5, "y": 0}
        yield {"feat": 0.5, "y": 1}
    for _ in range(8):
        yield {"feat": 3.0, "y": 1}
    for i in range(12):
        yield {"feat": 2.5 + 0.1 * (i % 3), "y": i % 2}


def test_live_tick_drift_retrain_and_recalibration(tmp_path):
    ring_path = tmp_path / "ticks.ring"
    model_path = tmp_path / "model.json"
    buffer_path = tmp_path / "data/live_ticks.parquet"
    trainer = OnlineTrainer(
        model_path=model_path,
        batch_size=4,
        tick_buffer_path=buffer_path,
    )
    trainer.drift_detector = PageHinkley(delta=0.0, threshold=0.05, min_samples=5)
    trainer.drift_baseline_min = 6
    trainer.drift_recent_min = 4
    trainer.psi_threshold = 0.01
    trainer.ks_threshold = 0.01
    asyncio.run(trainer.consume_ticks(_drift_stream(), ring_path))
    assert buffer_path.exists()
    df = pd.read_parquet(buffer_path)
    assert not df.empty
    assert trainer.drift_events >= 1
    assert trainer.drift_event_log
    assert any(evt.get("type") in {"sequential", "feature"} for evt in trainer.drift_event_log)
    assert trainer.calibrator is not None
    assert trainer.calibration_metadata.get("samples", 0) >= trainer.calibration_min_samples
    assert len(trainer.calib_scores) == trainer.calibration_metadata.get("samples")
    data = json.loads(model_path.read_text())
    assert data.get("online_drift_events")
    calib = data.get("calibration")
    assert calib and calib.get("samples", 0) == trainer.calibration_metadata.get("samples")


async def _stability_pre_drift_stream():
    for _ in range(12):
        yield {"feat": -1.0, "y": 0}
        yield {"feat": 1.0, "y": 1}


async def _stability_drift_only_stream():
    for _ in range(24):
        yield {"feat": 3.0, "y": 1}


def test_replay_buffer_stability_and_memory(tmp_path):
    replay_ring = tmp_path / "replay.ring"
    plain_ring = tmp_path / "plain.ring"
    replay_model = tmp_path / "replay_model.json"
    plain_model = tmp_path / "plain_model.json"

    trainer_replay = OnlineTrainer(
        model_path=replay_model,
        batch_size=8,
        replay_capacity=64,
        replay_sample_size=8,
    )
    trainer_plain = OnlineTrainer(
        model_path=plain_model,
        batch_size=8,
        replay_capacity=64,
        replay_sample_size=0,
    )

    asyncio.run(trainer_replay.consume_ticks(_stability_pre_drift_stream(), replay_ring))
    asyncio.run(trainer_plain.consume_ticks(_stability_pre_drift_stream(), plain_ring))

    asyncio.run(trainer_replay.consume_ticks(_stability_drift_only_stream(), replay_ring))
    asyncio.run(trainer_plain.consume_ticks(_stability_drift_only_stream(), plain_ring))

    eval_batch = [{"feat": -1.0, "y": 0}, {"feat": 1.0, "y": 1}] * 16
    X_replay, y_eval = trainer_replay._vectorise(eval_batch)
    acc_replay = float(np.mean(trainer_replay.clf.predict(X_replay) == y_eval))
    X_plain, y_eval_plain = trainer_plain._vectorise(eval_batch)
    acc_plain = float(np.mean(trainer_plain.clf.predict(X_plain) == y_eval_plain))
    assert acc_replay >= acc_plain

    assert len(trainer_replay.replay_buffer) <= trainer_replay.replay_buffer.capacity
    saved = json.loads(replay_model.read_text())
    state = saved.get("replay_buffer")
    assert state is not None
    assert len(state.get("items", [])) <= trainer_replay.replay_buffer.capacity
    labels = [int(item.get("y", -1)) for item in state.get("items", [])]
    assert any(lbl == 0 for lbl in labels)
    assert any(lbl == 1 for lbl in labels)

    restored = OnlineTrainer(
        model_path=replay_model,
        batch_size=8,
        replay_capacity=64,
        replay_sample_size=8,
    )
    assert len(restored.replay_buffer) == len(trainer_replay.replay_buffer)
