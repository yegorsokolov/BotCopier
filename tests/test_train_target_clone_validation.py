import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from botcopier.training.pipeline import (
    train,
    _HAS_OPTUNA,
    _HAS_TORCH,
    TabTransformer,
    TCNClassifier,
    _load_logs,
    _extract_features,
    predict_expected_value,
)


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
    assert "sharpe_ratio" in params["metrics"]
    assert "sortino_ratio" in params["metrics"]


def test_threshold_optimisation_changes_value(tmp_path: Path) -> None:
    rows = [
        "label,profit,hour,spread\n",
        "1,2,1,1.0\n",
        "0,-1,2,0.0\n",
        "1,2,3,1.1\n",
        "0,-0.5,4,0.1\n",
        "1,2,5,1.2\n",
        "0,-0.5,6,0.2\n",
    ]
    data = tmp_path / "trades_raw.csv"
    data.write_text("".join(rows))
    out_s = tmp_path / "out_s"
    train(data, out_s, threshold_objective="sharpe")
    thr_s = json.loads((out_s / "model.json").read_text())["session_models"]["asian"]["threshold"]
    out_so = tmp_path / "out_so"
    train(data, out_so, threshold_objective="sortino")
    thr_so = json.loads((out_so / "model.json").read_text())["session_models"]["asian"]["threshold"]
    assert thr_s != thr_so


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


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_tabtransformer_predictions(tmp_path):
    import torch

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
    train(data_file, out_dir, model_type="tabtransformer", epochs=5)
    model = json.loads((out_dir / "model.json").read_text())
    feature_names = model["feature_names"]
    state = {k: torch.tensor(v) for k, v in model["state_dict"].items()}
    tt = TabTransformer(len(feature_names))
    tt.load_state_dict(state)
    proc, fcols, _ = _load_logs(data_file)
    if not isinstance(proc, pd.DataFrame):
        proc = pd.concat(list(proc), ignore_index=True)
    proc, fcols, _, _ = _extract_features(proc, fcols)
    X_infer = proc[feature_names].to_numpy(dtype=float)
    c_low = np.array(model["clip_low"]) ; c_high = np.array(model["clip_high"])
    X_infer = np.clip(X_infer, c_low, c_high)
    mean = np.array(model["mean"]) ; std = np.array(model["std"])
    X_scaled = (X_infer - mean) / np.where(std == 0, 1, std)
    with torch.no_grad():
        preds = torch.sigmoid(
            tt(torch.tensor(X_scaled, dtype=torch.float32))
        ).numpy().ravel()
    assert preds.shape[0] == len(df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    acc = ((preds > 0.5) == y).mean()
    assert acc >= 0.5


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_tcn_predictions(tmp_path):
    import torch

    X, y = make_classification(
        n_samples=80,
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
    data_file = tmp_path / "trades_raw.csv"
    df.to_csv(data_file, index=False)
    out_dir = tmp_path / "out"
    train(data_file, out_dir, model_type="tcn", epochs=5, window=4)
    model = json.loads((out_dir / "model.json").read_text())
    feature_names = model["feature_names"]
    state = {k: torch.tensor(v) for k, v in model["state_dict"].items()}
    net = TCNClassifier(len(feature_names))
    net.load_state_dict(state)
    proc, fcols, _ = _load_logs(data_file)
    if not isinstance(proc, pd.DataFrame):
        proc = pd.concat(list(proc), ignore_index=True)
    proc, fcols, _, _ = _extract_features(proc, fcols)
    X_infer = proc[feature_names].to_numpy(dtype=float)
    c_low = np.array(model["clip_low"]) ; c_high = np.array(model["clip_high"])
    X_infer = np.clip(X_infer, c_low, c_high)
    mean = np.array(model["mean"]) ; std = np.array(model["std"])
    X_scaled = (X_infer - mean) / np.where(std == 0, 1, std)
    win = int(model["window"])
    seqs = [X_scaled[i - win : i].T for i in range(win, len(X_scaled))]
    X_seq = torch.tensor(np.stack(seqs), dtype=torch.float32)
    with torch.no_grad():
        preds = torch.sigmoid(net(X_seq)).numpy().ravel()
    assert preds.shape[0] == len(df) - win
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    acc = ((preds > 0.5) == y[win:]).mean()
    assert acc >= 0.5


def test_expected_value_pipeline_outputs_expected_profit(tmp_path: Path) -> None:
    rows = [
        "label,profit,hour,spread\n",
        "1,2,1,0.5\n",
        "0,-1,2,0.5\n",
        "1,3,3,0.5\n",
        "0,-0.5,4,0.5\n",
    ]
    data = tmp_path / "trades_raw.csv"
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    train(data, out_dir, expected_value=True)
    model = json.loads((out_dir / "model.json").read_text())
    params = next(iter(model["session_models"].values()))
    proc, fcols, _ = _load_logs(data)
    if not isinstance(proc, pd.DataFrame):
        proc = pd.concat(list(proc), ignore_index=True)
    proc, fcols, _, _ = _extract_features(proc, fcols)
    X = proc[model["feature_names"]].to_numpy(dtype=float)
    preds = predict_expected_value(model, X)
    mean = np.array(params["feature_mean"])
    std = np.array(params["feature_std"])
    X_scaled = (X - mean) / np.where(std == 0, 1, std)
    coef = np.array(params["coefficients"])
    logits = X_scaled @ coef + params["intercept"]
    prob = 1 / (1 + np.exp(-logits))
    pnl_coef = np.array(params["pnl_model"]["coefficients"])
    pnl = X_scaled @ pnl_coef + params["pnl_model"]["intercept"]
    expected_manual = prob * pnl
    assert np.allclose(preds, expected_manual)
