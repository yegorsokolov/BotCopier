from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from botcopier.strategy.dsl import (
    Add,
    And,
    Constant,
    Div,
    EMA,
    GT,
    LT,
    Mul,
    Or,
    Position,
    Price,
    SMA,
    StopLoss,
    Sub,
    deserialize,
    serialize,
)
from tests.property.strategies import price_series


# Strategy for generating random DSL expressions
expr_strategy = st.recursive(
    st.one_of(
        st.just(Price()),
        st.builds(SMA, st.integers(1, 3)),
        st.builds(EMA, st.integers(1, 3)),
        st.builds(Constant, st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
    ),
    lambda children: st.one_of(
        st.builds(Add, children, children),
        st.builds(Sub, children, children),
        st.builds(Mul, children, children),
        st.builds(Div, children, children),
        st.builds(GT, children, children),
        st.builds(LT, children, children),
        st.builds(And, children, children),
        st.builds(Or, children, children),
        st.builds(Position, children, st.floats(-5, 5, allow_nan=False, allow_infinity=False)),
        st.builds(StopLoss, children, st.floats(0, 5, allow_nan=False, allow_infinity=False)),
    ),
    max_leaves=10,
)


@given(expr_strategy, price_series())
@settings(max_examples=100)
def test_serialize_roundtrip(expr, prices):
    data = serialize(expr)
    rebuilt = deserialize(data)
    assert np.allclose(rebuilt.eval(prices), expr.eval(prices))
