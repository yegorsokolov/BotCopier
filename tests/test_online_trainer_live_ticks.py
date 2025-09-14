import asyncio

import pandas as pd

from botcopier.scripts.online_trainer import OnlineTrainer


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
