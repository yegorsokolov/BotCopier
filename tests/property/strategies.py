"""Hypothesis strategies for property tests."""
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.extra.pandas import column, data_frames, range_indexes


def trade_logs():
    """Generate synthetic trade log DataFrames."""
    return data_frames(
        columns=[
            column(
                "price",
                st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
            ),
            column(
                "spread",
                st.floats(0, 1e3, allow_nan=False, allow_infinity=False),
            ),
        ],
        index=range_indexes(min_size=1, max_size=50),
    )


def feature_matrices():
    """Generate random feature matrices."""
    shapes = array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10)
    return arrays(
        np.float64,
        shapes,
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    )


def model_parameters():
    """Generate model parameter dictionaries for scaling."""

    def build(size: int):
        return st.fixed_dictionaries(
            {
                "feature_mean": st.lists(
                    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    min_size=size,
                    max_size=size,
                ),
                "feature_std": st.lists(
                    st.floats(0.1, 1e3, allow_nan=False, allow_infinity=False),
                    min_size=size,
                    max_size=size,
                ),
            }
        )

    return st.integers(1, 10).flatmap(build)


def price_series():
    """Generate random price series for strategy testing."""

    return st.lists(
        st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=50,
    ).map(lambda vals: np.asarray(vals, dtype=float))
