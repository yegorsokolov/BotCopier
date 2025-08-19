import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_replay_outputs(tmp_path):
    decisions = pd.DataFrame(
        [
            {"event_id": 1, "probability": 0.2, "f1": 1.0, "profit": 1.0},
            {"event_id": 2, "probability": 0.8, "f1": -1.0, "profit": -1.0},
        ]
    )
    log_file = tmp_path / "decisions.csv"
    decisions.to_csv(log_file, index=False, sep=";")
    model = {
        "feature_names": ["f1"],
        "coefficients": [1.0],
        "intercept": 0.0,
        "threshold": 0.5,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)
    out_file = tmp_path / "divergences.csv"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    script = repo_root / "scripts" / "replay_decisions.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(log_file),
            str(model_file),
            "--weight",
            "2",
            "--output",
            str(out_file),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    df = pd.read_csv(out_file)
    assert set(df["event_id"]) == {1, 2}
    assert list(df["weight"]) == [2.0, 2.0]
