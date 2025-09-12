import numpy as np

from scripts.model_fitting import fit_logistic_regression, train_model


def test_fit_logistic_regression():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = fit_logistic_regression(X, y)
    preds = clf.predict(X)
    assert (preds == y).mean() >= 0.75


def test_train_model_half_life_weights(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    times = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ],
        dtype="datetime64[s]",
    )
    model_long = train_model(X, y, times, tmp_path / "long", half_life_days=1000)
    model_short = train_model(X, y, times, tmp_path / "short", half_life_days=0.5)

    def _predict(model, x):
        z = model["coefficients"][0] * x + model["intercept"]
        return 1 / (1 + np.exp(-z))

    proba_long = _predict(model_long, 3.0)
    proba_short = _predict(model_short, 3.0)
    assert proba_short > proba_long
    assert model_short["half_life_days"] == 0.5
    assert model_long["half_life_days"] == 1000


def test_vif_filtering_reduces_features_and_changes_coefficients(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 * 0.99 + rng.normal(scale=0.01, size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([x1, x2, x3])
    y = (x1 + x3 + rng.normal(scale=0.1, size=n) > 0).astype(int)
    times = np.arange(n, dtype="timedelta64[s]") + np.datetime64("2024-01-01")
    feats = ["x1", "x2", "x3"]

    model_no = train_model(
        X, y, times, tmp_path / "no_vif", feature_names=feats, vif_threshold=float("inf")
    )
    model_vif = train_model(
        X, y, times, tmp_path / "with_vif", feature_names=feats, vif_threshold=5.0
    )

    assert len(model_vif["coefficients"]) < len(model_no["coefficients"])
    assert not np.isclose(model_vif["intercept"], model_no["intercept"])
