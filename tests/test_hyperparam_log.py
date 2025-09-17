import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("optuna")

from botcopier.training.pipeline import run_optuna


def test_hyperparam_csv_contains_best_trial(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 4))
    weights = np.array([0.8, -0.5, 0.3, 0.2])
    logits = X @ weights + rng.normal(scale=0.2, size=X.shape[0])
    y = (logits > 0).astype(int)
    profits = np.where(y == 1, rng.uniform(0.5, 1.5, size=y.size), rng.uniform(-1.0, 0.2, size=y.size))
    df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4"])
    df["label"] = y
    df["profit"] = profits
    data_file = tmp_path / "trades.csv"
    df.to_csv(data_file, index=False)

    out_dir = tmp_path / "study"
    csv_path = out_dir / "hyperparams.csv"
    model_path = out_dir / "model.json"

    study = run_optuna(
        n_trials=3,
        csv_path=csv_path,
        model_json_path=model_path,
        settings_overrides={"data_dir": data_file, "out_dir": out_dir},
        model_types=["logreg"],
        feature_flags={"vol_weight": [False, True]},
        train_kwargs={"n_splits": 2, "cv_gap": 1},
    )

    assert csv_path.exists(), "hyperparams.csv should be created"
    assert model_path.exists(), "Final model should be written"

    data = json.loads(model_path.read_text())
    metadata = data.get("metadata", {})
    assert metadata["hyperparam_log"] == "hyperparams.csv"
    assert "selected_trial" in metadata
    selected = metadata["selected_trial"]

    df = pd.read_csv(csv_path)
    best_row = df[df["trial"] == selected["number"]].iloc[0]
    assert best_row["profit"] == pytest.approx(selected["profit"])
    assert best_row["sharpe"] == pytest.approx(selected["sharpe"])
    assert best_row["max_drawdown"] == pytest.approx(selected["max_drawdown"])
    assert best_row["var_95"] == pytest.approx(selected["var_95"])

    search_params = selected.get("search_params", {})
    assert "seed" in search_params
    assert search_params.get("model_type", "logreg") == "logreg"
    assert "vol_weight" in search_params

    risk_metrics = data.get("risk_metrics", {})
    assert risk_metrics.get("max_drawdown") == pytest.approx(selected["max_drawdown"])
    assert risk_metrics.get("var_95") == pytest.approx(selected["var_95"])

    hyperopt = metadata.get("hyperparameter_optimization", {})
    assert hyperopt.get("n_trials") == len(
        [t for t in study.trials if t.state.name == "COMPLETE"]
    )
    best_meta = hyperopt.get("best_trial", {})
    assert best_meta.get("number") == selected["number"]
    assert "artifact_dir" in best_meta
    artifact_dir = out_dir / best_meta["artifact_dir"]
    assert artifact_dir.exists()
