import numpy as np

from botcopier.strategy.dsl import (
    GT,
    SMA,
    Add,
    Constant,
    Position,
    Price,
    Sub,
    backtest,
    deserialize,
    serialize,
)
from botcopier.strategy.search import search_strategy


def test_dsl_compile_and_serialize() -> None:
    prices = np.linspace(1.0, 100.0, 100)
    expr = Position(GT(Add(SMA(3), Constant(1)), Sub(SMA(5), Constant(1))))
    compiled = expr.compile()
    out = compiled(prices)
    assert out.shape == prices.shape
    assert backtest(prices, expr) > 0

    data = serialize(expr)
    rebuilt = deserialize(data)
    assert np.allclose(rebuilt.eval(prices), expr.eval(prices))


def test_search_strategy_profitable() -> None:
    prices = np.linspace(1.0, 100.0, 200)
    baseline = backtest(prices, Position(GT(Price(), Price())))
    expr, ret, risk = search_strategy(prices, n_samples=10, seed=0)
    assert ret >= baseline
    assert risk >= 0
    assert backtest(prices, expr) == ret
