import pytest

np = pytest.importorskip("numpy")

from botcopier.models.registry import get_model
from botcopier.scripts.evaluation import search_decision_threshold


def test_ensemble_voting_improves_profit() -> None:
    rng = np.random.default_rng(0)
    n_linear = 200
    n_xor = 200

    X_linear = rng.normal(size=(n_linear, 2))
    y_linear = (X_linear[:, 0] > 0).astype(float)
    profits_linear = np.where(y_linear == 1.0, 2.0, -1.0)

    X_xor = rng.integers(0, 2, size=(n_xor, 2)).astype(float)
    y_xor = (X_xor[:, 0] != X_xor[:, 1]).astype(float)
    profits_xor = np.where(y_xor == 1.0, 2.0, -1.0)

    X = np.vstack([X_linear, X_xor])
    y = np.concatenate([y_linear, y_xor])
    profits = np.concatenate([profits_linear, profits_xor])

    perm = rng.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    profits = profits[perm]

    split = int(0.8 * X.shape[0])
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    profits_val = profits[split:]

    _, log_predict = get_model("logreg")(X_train, y_train)
    log_probs = log_predict(X_val)
    _, log_metrics = search_decision_threshold(
        y_val, log_probs, profits_val, objective="profit"
    )
    log_profit = log_metrics.get("profit", 0.0)

    _, gb_predict = get_model("gradient_boosting")(
        X_train, y_train, random_state=0, max_depth=2
    )
    gb_probs = gb_predict(X_val)
    _, gb_metrics = search_decision_threshold(
        y_val, gb_probs, profits_val, objective="profit"
    )
    gb_profit = gb_metrics.get("profit", 0.0)

    _, ens_predict = get_model("ensemble_voting")(
        X_train,
        y_train,
        random_state=0,
        gb_params={"max_depth": 2},
    )
    ens_probs = ens_predict(X_val)
    _, ens_metrics = search_decision_threshold(
        y_val, ens_probs, profits_val, objective="profit"
    )
    ens_profit = ens_metrics.get("profit", 0.0)

    assert ens_profit >= log_profit
    assert ens_profit >= gb_profit
