import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.online_trainer import OnlineTrainer


def test_online_trainer_updates(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "model.json"
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))

    trainer = OnlineTrainer(model_path=model_path, batch_size=2, run_generator=True)
    batch = [
        {"a": 1.0, "b": 0.0, "y": 1},
        {"a": 0.0, "b": 1.0, "y": 0},
    ]
    trainer.update(batch)

    data = json.loads(model_path.read_text())
    assert set(data["feature_names"]) == {"a", "b"}
    assert len(calls) == 1  # generation triggered

    trainer2 = OnlineTrainer(model_path=model_path, batch_size=1, run_generator=False)
    before = trainer2.clf.coef_.copy()
    trainer2.update([{"a": 1.0, "b": 1.0, "y": 1}])
    after = trainer2.clf.coef_.copy()
    assert not np.array_equal(before, after)

