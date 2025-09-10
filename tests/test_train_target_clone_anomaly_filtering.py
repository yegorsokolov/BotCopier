import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from botcopier.training.pipeline import train


def test_anomaly_filtering_reduces_rows_and_changes_coefficients(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=42,
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

    out1 = tmp_path / "no_filter"
    train(data_file, out1)
    model1 = json.loads((out1 / "model.json").read_text())
    coeffs1 = model1["session_models"]["asian"]["coefficients"]

    out2 = tmp_path / "with_filter"
    train(data_file, out2, anomaly_threshold=0.9)
    model2 = json.loads((out2 / "model.json").read_text())
    coeffs2 = model2["session_models"]["asian"]["coefficients"]

    assert model2["training_rows"] < model1["training_rows"]
    assert any(abs(a - b) > 1e-6 for a, b in zip(coeffs1, coeffs2))
