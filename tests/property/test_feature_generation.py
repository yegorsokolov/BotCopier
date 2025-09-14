from hypothesis import assume, given, strategies as st
import math

from botcopier.scripts.features import _sma, _atr, _bollinger


@st.composite
def value_series(draw):
    size = draw(st.integers(min_value=1, max_value=50))
    return draw(
        st.lists(
            st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )


@given(value_series(), st.integers(1, 50))
def test_sma_within_bounds(values, window):
    sma = _sma(values, window)
    assert math.isfinite(sma)
    assert min(values) <= sma <= max(values)


@given(value_series(), st.integers(1, 50))
def test_atr_non_negative(values, window):
    assume(len(values) > 1)
    atr = _atr(values, window)
    assert math.isfinite(atr)
    assert atr >= 0


@given(value_series(), st.integers(1, 50))
def test_bollinger_order(values, window):
    upper, mid, lower = _bollinger(values, window)
    assert upper >= mid >= lower
    assert math.isfinite(upper) and math.isfinite(mid) and math.isfinite(lower)
