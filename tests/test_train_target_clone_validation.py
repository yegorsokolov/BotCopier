import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from scripts.train_target_clone import train, _HAS_OPTUNA


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


def test_volatility_weighting_changes_coefficients(tmp_path):
    X, y = make_classification(
        n_samples=200,
        n_features=12,
        n_informative=5,
        n_redundant=0,
        random_state=0,
    )
    cols = [
        "price",
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
    df["price"] = 100 + df["price"].cumsum()
    df["label"] = y
    data_file = tmp_path / "trades_raw.csv"
    df.to_csv(data_file, index=False)

    out_dir1 = tmp_path / "out1"
    train(data_file, out_dir1)
    model1 = json.loads((out_dir1 / "model.json").read_text())
    coeffs1 = model1["session_models"]["asian"]["coefficients"]

    out_dir2 = tmp_path / "out2"
    train(data_file, out_dir2, use_volatility_weight=True)
    model2 = json.loads((out_dir2 / "model.json").read_text())
    coeffs2 = model2["session_models"]["asian"]["coefficients"]

    assert any(abs(a - b) > 1e-6 for a, b in zip(coeffs1, coeffs2))


def test_profit_weighting_changes_coefficients(tmp_path):
    X, y = make_classification(
        n_samples=200,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=1,
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
    df["profit"] = np.where(y == 1, 2.0, 0.5)
    data_file = tmp_path / "trades_raw.csv"
    df.to_csv(data_file, index=False)

    out_dir1 = tmp_path / "out1"
    train(data_file, out_dir1)
    model1 = json.loads((out_dir1 / "model.json").read_text())
    coeffs1 = model1["session_models"]["asian"]["coefficients"]

    out_dir2 = tmp_path / "out2"
    train(data_file, out_dir2, use_profit_weight=True)
    model2 = json.loads((out_dir2 / "model.json").read_text())
    coeffs2 = model2["session_models"]["asian"]["coefficients"]

    assert any(abs(a - b) > 1e-6 for a, b in zip(coeffs1, coeffs2))


@pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")
def test_optuna_cross_validation_respects_fold_count(tmp_path):
    X, y = make_classification(
        n_samples=60,
        n_features=11,
        n_informative=5,
        n_redundant=0,
        random_state=42,
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
    df["profit"] = (y * 2 - 1).astype(float)
    data_file = tmp_path / "trades_raw.csv"
    df.to_csv(data_file, index=False)

    out_dir2 = tmp_path / "out2"
    train(data_file, out_dir2, optuna_trials=1, optuna_folds=2)
    model2 = json.loads((out_dir2 / "model.json").read_text())
    folds2 = model2["optuna_best_fold_scores"]
    assert len(folds2) == 2

    out_dir3 = tmp_path / "out3"
    train(data_file, out_dir3, optuna_trials=1, optuna_folds=3)
    model3 = json.loads((out_dir3 / "model.json").read_text())
    folds3 = model3["optuna_best_fold_scores"]
    assert len(folds3) == 3

    assert model2["optuna_best_score"] != model3["optuna_best_score"]


def test_synthetic_rows_expand_dataset_and_metrics(tmp_path):
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,spread,hour\n",
        "0,1.0,1.0,0\n",
        "1,1.1,1.0,1\n",
        "0,1.2,1.0,2\n",
        "1,1.3,1.0,3\n",
    ]
    data.write_text("".join(rows))
    out1 = tmp_path / "out1"
    train(data, out1)
    model1 = json.loads((out1 / "model.json").read_text())

    synth = tmp_path / "synthetic_prices.csv"
    synth.write_text("price,hour\n1.4,4\n1.5,5\n1.6,6\n")
    out2 = tmp_path / "out2"
    train(data, out2, synthetic_data=synth, synthetic_weight=0.5)
    model2 = json.loads((out2 / "model.json").read_text())

    sm1 = model1["synthetic_metrics"]
    sm2 = model2["synthetic_metrics"]
    assert sm1["synthetic"] == 0
    assert sm2["synthetic"] > 0
    assert sm2["all"] > sm1["all"]
    assert model1["cv_accuracy"] != model2["cv_accuracy"]
