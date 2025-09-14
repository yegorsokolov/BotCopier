import json
from pathlib import Path

from scripts.online_trainer import OnlineTrainer


def test_online_trainer_persists_meta(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"meta": {"weights": [0.0, 0.0]}}))
    trainer = OnlineTrainer(model_path=model_path, batch_size=2, run_generator=False)
    batch = [
        {"f0": 0.1, "f1": 0.2, "y": 1},
        {"f0": -0.1, "f1": -0.2, "y": 0},
    ]
    assert trainer.update(batch)
    trainer._save()
    data = json.loads(model_path.read_text())
    assert "meta" in data and "weights" in data["meta"]
    assert data["meta"]["weights"] != [0.0, 0.0]
    assert len(data.get("adaptation_log", [])) >= 1
