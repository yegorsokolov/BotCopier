import math
import os

import numpy as np
import pandas as pd
import pytest

from botcopier.features import engineering as feature_engineering
from botcopier.scripts import evaluation as eval_module


def _python_abs_drawdown(values):
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for val in values:
        cumulative += float(val)
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _python_risk(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _python_budget_utilisation(values, budget):
    used = sum(abs(v) for v in values)
    if budget <= 0:
        return used
    return used / budget


def _python_order_type_compliance(order_types, allowed):
    if not order_types:
        return 1.0
    allowed_set = {a.lower() for a in allowed}
    cleaned = [o.strip().lower() for o in order_types]
    compliant = sum(1 for val in cleaned if val in allowed_set)
    return compliant / len(cleaned)


def _python_var95(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = int(0.05 * len(ordered))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _python_volatility_spikes(values):
    values = list(values)
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return 0
    return sum(1 for v in values if abs(v - mean) > 3 * std)


def _python_slippage_stats(values):
    values = list(values)
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def _assert_nested_close(left, right, atol=1e-12):
    assert left.keys() == right.keys()
    for key, lhs in left.items():
        rhs = right[key]
        if isinstance(lhs, dict):
            _assert_nested_close(lhs, rhs, atol=atol)
        elif isinstance(lhs, (list, tuple)):
            np.testing.assert_allclose(lhs, rhs, atol=atol, rtol=0)
        elif isinstance(lhs, np.ndarray):
            np.testing.assert_allclose(lhs, rhs, atol=atol, rtol=0)
        else:
            if lhs is None:
                assert rhs is None
            elif isinstance(lhs, float) and math.isnan(lhs):
                assert isinstance(rhs, float) and math.isnan(rhs)
            else:
                assert pytest.approx(lhs, abs=atol) == rhs


def test_merge_feature_frames_vectorized_matches_loop():
    base = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    plugin_one = pd.DataFrame({"c": [5, 6]})
    plugin_two = pd.DataFrame({"b": [7, 8], "d": [9, 10]})
    plugin_results = [
        ("one", plugin_one, ["c"]),
        ("two", plugin_two, ["b", "d"]),
    ]

    merged, features = feature_engineering._merge_feature_frames(
        base, ["a", "b"], plugin_results
    )

    expected_df = pd.concat([base, plugin_one, plugin_two], axis=1)
    expected_df = expected_df.loc[:, ~expected_df.columns.duplicated(keep="last")]
    expected_features = ["a", "b", "c", "d"]

    pd.testing.assert_frame_equal(merged, expected_df)
    assert features == expected_features


def test_financial_metrics_vectorization_matches_baseline():
    returns = np.array([1.2, -0.5, 0.3, -1.0, 0.8], dtype=float)
    order_types = ["BUY", "SELL", "BUY", "SELL", "BUY"]
    allowed = ["BUY"]
    slippage = np.array([0.1, -0.2, 0.05], dtype=float)

    assert eval_module._abs_drawdown(returns) == pytest.approx(
        _python_abs_drawdown(returns)
    )
    assert eval_module._risk(returns) == pytest.approx(_python_risk(returns))
    assert eval_module._budget_utilisation(returns, 3.0) == pytest.approx(
        _python_budget_utilisation(returns, 3.0)
    )
    assert eval_module._order_type_compliance(order_types, allowed) == pytest.approx(
        _python_order_type_compliance(order_types, allowed)
    )
    assert eval_module._var_95(returns) == pytest.approx(_python_var95(returns))
    assert eval_module._volatility_spikes(returns) == _python_volatility_spikes(
        returns
    )
    assert eval_module._slippage_stats(slippage) == pytest.approx(
        _python_slippage_stats(slippage)
    )


def _bootstrap_baseline(y, probs, returns, n_boot):
    rng = np.random.default_rng(0)
    keys = [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "var_95",
        "profit",
    ]
    collected = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        metrics = eval_module._classification_metrics(y[idx], probs[idx], returns[idx])
        for key in keys:
            val = metrics.get(key)
            if val is not None:
                collected[key].append(float(val))
    results = {}
    for key, values in collected.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        low, high = np.quantile(arr, [0.025, 0.975])
        results[key] = {"mean": mean, "low": float(low), "high": float(high)}
    return results


def test_bootstrap_metrics_matches_baseline(tmp_path):
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=32)
    probs = rng.random(32)
    returns = rng.normal(0, 1, size=32)

    expected = _bootstrap_baseline(y, probs, returns, n_boot=32)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = eval_module.bootstrap_metrics(
            y, probs, returns, n_boot=32, n_jobs=1
        )
    finally:
        os.chdir(cwd)

    _assert_nested_close(expected, result)


def test_bootstrap_metrics_parallel_deterministic(tmp_path):
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=24)
    probs = rng.random(24)
    returns = rng.standard_normal(24)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        seq = eval_module.bootstrap_metrics(y, probs, returns, n_boot=48, n_jobs=1)
        par = eval_module.bootstrap_metrics(y, probs, returns, n_boot=48, n_jobs=3)
    finally:
        os.chdir(cwd)

    _assert_nested_close(seq, par)


def _search_baseline(
    y_true,
    probas,
    profits,
    *,
    objective="profit",
    threshold_grid=None,
    metric_names=None,
    max_drawdown=None,
    var_limit=None,
):
    candidates = eval_module._candidate_thresholds(probas, threshold_grid)
    objective_metric_map = {
        "profit": "profit",
        "net_profit": "profit",
        "sharpe": "sharpe_ratio",
        "sortino": "sortino_ratio",
    }
    metric_key = objective_metric_map.get(objective, objective)
    best_threshold = float(candidates[0])
    best_metrics = None
    best_score = -np.inf
    for thr in candidates:
        returns = profits * (probas >= thr)
        metrics = eval_module._classification_metrics(
            y_true,
            probas,
            returns,
            selected=metric_names,
            threshold=float(thr),
        )
        metrics.setdefault("max_drawdown", float(eval_module._abs_drawdown(list(returns))))
        metrics.setdefault("var_95", float(eval_module._var_95(list(returns))))
        metrics["threshold"] = float(thr)
        metrics["threshold_objective"] = objective

        if max_drawdown is not None and metrics["max_drawdown"] > max_drawdown:
            continue
        if var_limit is not None and metrics["var_95"] > var_limit:
            continue

        obj_val = metrics.get(metric_key)
        if isinstance(obj_val, (int, float)) and not math.isnan(float(obj_val)):
            score = float(obj_val)
        else:
            score = -np.inf

        if score > best_score or (
            np.isfinite(score)
            and np.isclose(score, best_score)
            and float(thr) < best_threshold
        ):
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    if best_metrics is None:
        limit_desc = []
        if max_drawdown is not None:
            limit_desc.append("max_drawdown")
        if var_limit is not None:
            limit_desc.append("var_95")
        if limit_desc:
            raise ValueError(
                "No decision threshold satisfies risk limits: "
                + ", ".join(limit_desc)
            )
        fallback = float(candidates[0])
        returns = profits * (probas >= fallback)
        best_metrics = eval_module._classification_metrics(
            y_true,
            probas,
            returns,
            selected=metric_names,
            threshold=fallback,
        )
        best_metrics.setdefault(
            "max_drawdown", float(eval_module._abs_drawdown(list(returns)))
        )
        best_metrics.setdefault("var_95", float(eval_module._var_95(list(returns))))
        best_metrics["threshold"] = fallback
        best_metrics["threshold_objective"] = objective
        best_threshold = fallback

    return best_threshold, best_metrics


def test_search_decision_threshold_matches_baseline():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=40)
    probas = rng.random(40)
    profits = rng.normal(0, 1, size=40)

    expected_thr, expected_metrics = _search_baseline(
        y_true,
        probas,
        profits,
        objective="sharpe",
        threshold_grid=[0.2, 0.4, 0.6],
        metric_names=["sharpe_ratio", "max_drawdown"],
        max_drawdown=2.0,
    )

    thr, metrics = eval_module.search_decision_threshold(
        y_true,
        probas,
        profits,
        objective="sharpe",
        threshold_grid=[0.2, 0.4, 0.6],
        metric_names=["sharpe_ratio", "max_drawdown"],
        max_drawdown=2.0,
        n_jobs=2,
    )

    assert thr == pytest.approx(expected_thr)
    _assert_nested_close(expected_metrics, metrics)


def test_search_decision_threshold_parallel_deterministic():
    rng = np.random.default_rng(21)
    y_true = rng.integers(0, 2, size=36)
    probas = rng.random(36)
    profits = rng.normal(0, 1, size=36)

    sequential = eval_module.search_decision_threshold(
        y_true, probas, profits, objective="profit", n_jobs=1
    )
    parallel = eval_module.search_decision_threshold(
        y_true, probas, profits, objective="profit", n_jobs=4
    )

    assert sequential[0] == pytest.approx(parallel[0])
    _assert_nested_close(sequential[1], parallel[1])
