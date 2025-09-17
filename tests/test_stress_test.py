from botcopier.scripts.stress_test import run_stress_tests, summarise_stress_results


def test_stress_harness_produces_metrics():
    returns = [0.1, 0.05, -0.02, 0.03, 0.04]
    order_types = ["market", "limit", "limit", "market", "limit"]

    results = run_stress_tests(returns, order_types=order_types)
    assert set(results.keys()) == {
        "baseline",
        "shock",
        "volatility_regime",
        "liquidity_drought",
    }

    liquidity = results["liquidity_drought"]
    assert liquidity.limit_fill_rate is not None
    assert liquidity.limit_fill_rate < 1.0

    summary = summarise_stress_results(results)
    assert summary["stress_pnl_min"] <= results["baseline"].pnl
    assert summary["stress_drawdown_max"] >= results["baseline"].max_drawdown
    assert (
        summary["stress_limit_fill_rate_min"]
        <= results["liquidity_drought"].limit_fill_rate
    )
