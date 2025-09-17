import os
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluation import bootstrap_metrics, evaluate_model, evaluate_strategy


def test_evaluate_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(max_iter=100).fit(X, y)
    metrics = evaluate_model(clf, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert "reliability_curve" in metrics
    assert metrics["max_drawdown"] == 0.0
    assert metrics["var_95"] == 1.0


def test_bootstrap_interval_shrinks(tmp_path):
    rng = np.random.default_rng(0)

    n_small = 50
    probs_small = rng.random(n_small)
    y_small = rng.binomial(1, probs_small)
    returns_small = rng.normal(0, 1, n_small)

    n_large = 2000
    probs_large = rng.random(n_large)
    y_large = rng.binomial(1, probs_large)
    returns_large = rng.normal(0, 1, n_large)

    old_cwd = os.getcwd()
    metrics_file = tmp_path / "metrics.json"
    os.chdir(tmp_path)
    try:
        metrics_small = bootstrap_metrics(
            y_small, probs_small, returns_small, n_boot=100
        )
        metrics_large = bootstrap_metrics(
            y_large, probs_large, returns_large, n_boot=100
        )
    finally:
        os.chdir(old_cwd)
    saved = json.loads(metrics_file.read_text())

    for key in ["accuracy", "brier_score", "sharpe_ratio"]:
        width_small = metrics_small[key]["high"] - metrics_small[key]["low"]
        width_large = metrics_large[key]["high"] - metrics_large[key]["low"]
        assert width_large < width_small

    for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "var_95"]:
        assert key in saved
        assert set(saved[key].keys()) == {"mean", "low", "high"}


def test_evaluate_strategy_includes_stress_metrics():
    returns = [0.12, -0.05, 0.08, 0.03]
    order_types = ["market", "limit", "limit", "market"]
    metrics = evaluate_strategy(
        returns,
        order_types,
        [],
        budget=10.0,
        allowed_order_types=["market", "limit"],
    )
    assert "stress_tests" in metrics
    assert "stress_summary" in metrics
    baseline = metrics["stress_tests"]["baseline"]
    shock = metrics["stress_tests"]["shock"]
    assert shock["pnl"] <= baseline["pnl"]
    summary = metrics["stress_summary"]
    assert summary["stress_pnl_min"] <= baseline["pnl"]
