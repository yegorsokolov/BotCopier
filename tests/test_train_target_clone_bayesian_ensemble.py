import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from scripts.train_target_clone import train


def test_bayesian_ensemble_metrics(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=80,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=0,
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
    train(data_file, out_dir, bayesian_ensembles=3)

    model = json.loads((out_dir / "model.json").read_text())
    bayes = model["bayesian_ensemble"]
    assert bayes["size"] == 3
    assert bayes["variance"] >= 0.0
    assert "roc_auc" in bayes["metrics"]["single"]
    assert "brier" in bayes["metrics"]["ensemble"]

