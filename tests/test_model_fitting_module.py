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
