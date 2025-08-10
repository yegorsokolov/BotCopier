import json
from pathlib import Path

from scripts.online_trainer import OnlineTrainer


def test_online_trainer_updates(tmp_path: Path):
    save_path = tmp_path / "model_online.json"
    trainer = OnlineTrainer(save_path=save_path, save_interval=0)
    events = [
        {"event_id": 1, "features": {"a": 1.0, "b": 0.0}, "y": 1},
        {"event_id": 2, "features": {"a": 0.0, "b": 1.0}, "y": 0},
    ]
    for e in events:
        trainer.process_event(e)
    trainer.save_model()
    data = json.loads(save_path.read_text())
    assert data["last_event_id"] == 2
    assert any(abs(c) > 0 for c in data["coefficients"])

    # Reload and continue training
    trainer2 = OnlineTrainer(save_path=save_path, save_interval=0)
    before = dict(trainer2.model._weights)
    trainer2.process_event({"event_id": 3, "features": {"a": 1.0, "b": 1.0}, "y": 1})
    trainer2.save_model()
    after = dict(trainer2.model._weights)
    assert after["a"] != before.get("a", 0.0) or after["b"] != before.get("b", 0.0)
