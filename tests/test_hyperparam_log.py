import json
from pathlib import Path

import pandas as pd
import pytest

from train_target_clone import run_optuna


def test_hyperparam_csv_contains_best_trial(tmp_path: Path):
    csv_path = tmp_path / "hyperparams.csv"
    model_path = tmp_path / "model.json"

    study = run_optuna(n_trials=5, csv_path=csv_path, model_json_path=model_path)

    assert csv_path.exists(), "hyperparams.csv should be created"

    data = json.loads(model_path.read_text())
    assert data["metadata"]["hyperparam_log"] == "hyperparams.csv"

    df = pd.read_csv(csv_path)
    best_number = data["metadata"]["best_trial"]["number"]
    best_value = data["metadata"]["best_trial"]["value"]
    best_row = df[df["trial"] == best_number].iloc[0]
    assert best_row["value"] == pytest.approx(best_value)
