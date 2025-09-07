import json
from pathlib import Path

import pandas as pd
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
