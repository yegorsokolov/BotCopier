import numpy as np
from hypothesis import given, strategies as st

from botcopier.strategy.dsl import Constant, Position, StopLoss
from tests.property.strategies import price_series


@given(price_series(), st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False))
def test_stop_loss_resets_on_large_drop(prices, limit):
    expr = StopLoss(Position(Constant(1.0)), limit)
    positions = expr.eval(prices)
    returns = np.diff(prices, prepend=prices[0])
    for pos, ret in zip(positions, returns):
        if ret < -abs(limit):
            assert pos == 0.0
        else:
            assert pos == 1.0
