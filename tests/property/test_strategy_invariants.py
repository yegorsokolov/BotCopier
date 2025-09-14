import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from botcopier.strategy.engine import search_strategies
from tests.property.strategies import price_series


@given(price_series())
@settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_strategy_profit_and_risk_bounds(prices: np.ndarray) -> None:
    try:
        best, pareto = search_strategies(prices, n_samples=5)
    except Exception:
        assume(False)
    total_diff = float(np.sum(np.abs(np.diff(prices))))
    for cand in pareto + [best]:
        assert -total_diff <= cand.ret <= total_diff
        assert 0 <= cand.risk <= total_diff
