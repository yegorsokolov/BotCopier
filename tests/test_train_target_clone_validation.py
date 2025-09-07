import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from scripts.train_target_clone import train


def test_voting_ensemble_improves_accuracy(tmp_path):
    X, y = make_classification(
        n_samples=200,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=0,
        flip_y=0.4,
    )
    cols = [
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
    ]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    data_file = tmp_path / "trades_raw.csv"
    df.to_csv(data_file, index=False)
    out_dir = tmp_path / "out"
    train(data_file, out_dir, ensemble="voting")
    model = json.loads((out_dir / "model.json").read_text())
    ensemble = model["ensemble"]
    assert ensemble["estimators"] == ["logreg", "gboost"]
    assert ensemble["accuracy"] >= max(ensemble["base_accuracies"].values())


def test_threshold_selected_by_profit(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,profit,hour,spread\n",
        "1,2,1,1.0\n",
        "0,1,2,0.0\n",
        "1,2,3,1.1\n",
        "0,1,4,0.1\n",
        "1,2,5,1.2\n",
        "0,1,6,0.2\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    params = model["session_models"]["asian"]
    assert params["threshold"] == pytest.approx(0.0)
    assert params["metrics"]["profit"] == pytest.approx(1.5)
