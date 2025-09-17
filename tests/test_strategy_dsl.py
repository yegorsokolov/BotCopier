import numpy as np

from botcopier.strategy.dsl import (
    ATR,
    BollingerBand,
    CrossPrice,
    GT,
    RSI,
    RollingVolatility,
    SMA,
    Add,
    Constant,
    Position,
    Price,
    TrailingStop,
    Sub,
    backtest,
    deserialize,
    serialize,
)
from botcopier.strategy.search import search_strategy


def test_dsl_compile_and_serialize() -> None:
    base = np.linspace(1.0, 100.0, 120)
    peer = base * 1.01
    context = {"base": base, "peer": peer}
    expr = TrailingStop(
        Position(
            GT(
                Add(RSI(6), BollingerBand(8, 1.4, "upper")),
                Add(CrossPrice("peer"), ATR(4)),
            ),
            size=0.6,
        ),
        lookback=7,
        buffer=0.4,
    )
    compiled = expr.compile()
    out = compiled(context)
    assert out.shape == base.shape
    pnl = backtest(context, expr)
    assert np.isfinite(pnl)

    data = serialize(expr)
    rebuilt = deserialize(data)
    assert np.allclose(rebuilt.eval(context), expr.eval(context))


def test_search_strategy_profitable() -> None:
    base = np.linspace(1.0, 100.0, 200)
    peer = base * 1.02
    context = {"base": base, "peer": peer}
    baseline = backtest(context, Position(GT(Price(), Price())))
    expr, ret, risk = search_strategy(context, n_samples=10, seed=0)
    assert ret >= baseline
    assert np.isfinite(risk) and risk >= 0
    compiled = expr.compile()
    assert compiled(context).shape == base.shape
    assert np.isclose(backtest(context, expr), ret)


def test_search_strategy_handles_short_series() -> None:
    prices = np.array([1.0, 1.0])
    expr, ret, risk = search_strategy(prices, n_samples=5, seed=1)
    assert expr.eval(prices).shape == prices.shape
    assert np.isfinite(ret)
    assert np.isfinite(risk)


def test_new_indicators_eval() -> None:
    base = np.linspace(10.0, 20.0, 50)
    peer = base[::-1] + 5.0
    context = {"base": base, "peer": peer}
    indicators = [
        RSI(5),
        ATR(5),
        BollingerBand(6, 2.0, "upper"),
        BollingerBand(6, 2.0, "lower"),
        RollingVolatility(4),
        CrossPrice("peer", normalize=False),
    ]
    for node in indicators:
        values = node.eval(context)
        assert values.shape == base.shape
        assert np.all(np.isfinite(values))

    trailing = TrailingStop(Position(Constant(1.0)), lookback=5, buffer=0.5)
    out = trailing.eval(context)
    assert out.shape == base.shape
