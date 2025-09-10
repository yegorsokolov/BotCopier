import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from botcopier.training.pipeline import train


def test_noise_filtering_reduces_dataset_size_and_changes_metrics(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=0,
        flip_y=0.3,
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

    out_dir1 = tmp_path / "no_filter"
    train(data_file, out_dir1)
    model1 = json.loads((out_dir1 / "model.json").read_text())

    out_dir2 = tmp_path / "with_filter"
    train(data_file, out_dir2, filter_noise=True)
    model2 = json.loads((out_dir2 / "model.json").read_text())

    assert model2["training_rows"] < model1["training_rows"]
    assert abs(model1["cv_accuracy"] - model2["cv_accuracy"]) > 1e-6
