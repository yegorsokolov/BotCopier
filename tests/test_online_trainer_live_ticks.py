import json
import asyncio

from botcopier.scripts.online_trainer import OnlineTrainer
from botcopier.scripts.shm_ring import ShmRing, TRADE_MSG


async def _tick_stream():
    for _ in range(32):
        yield {"feat": 0.0, "y": 0}
    for _ in range(32):
        yield {"feat": 10.0, "y": 1}


def test_tick_stream_updates_and_drift(tmp_path):
    ring_path = tmp_path / "ticks.ring"
    model_path = tmp_path / "model.json"
    trainer = OnlineTrainer(model_path=model_path, batch_size=16)
    if trainer.drift_detector:
        trainer.drift_detector.threshold = 0.01
        trainer.drift_detector.min_samples = 1
        trainer.drift_detector.delta = 0.0
    asyncio.run(trainer.consume_ticks(_tick_stream(), ring_path))
    ring = ShmRing.open(str(ring_path))
    count = 0
    while True:
        msg = ring.pop()
        if msg is None:
            break
        mtype, payload = msg
        if mtype == TRADE_MSG:
            count += 1
        bytes(payload)
        payload = None
    ring.close()
    assert count >= 64
    assert model_path.exists()
    data = json.loads(model_path.read_text())
    assert "coefficients" in data
    assert trainer.drift_events > 0
