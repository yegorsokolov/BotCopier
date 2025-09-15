"""Property-based tests for core invariants."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from hypothesis import HealthCheck, assume, given, settings
from sklearn.preprocessing import StandardScaler

from botcopier.data.loading import _compute_meta_labels
from botcopier.scripts.model_fitting import scale_features
from tests.property.strategies import feature_matrices, model_parameters, trade_logs


@given(trade_logs())
@settings(max_examples=25)
def test_trade_log_samples_are_finite(df) -> None:
    numeric = df.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()
    assert not df.isna().any().any()
    if "spread" in df.columns:
        assert (df["spread"] >= 0).all()


@given(feature_matrices())
@settings(max_examples=25)
def test_generated_feature_matrices_are_finite(X: np.ndarray) -> None:
    assert np.isfinite(X).all()
    assert not np.isnan(X).any()


@given(feature_matrices())
@settings(max_examples=20)
def test_feature_scaling_roundtrip(X: np.ndarray) -> None:
    scaler = StandardScaler()
    scaled = scale_features(scaler, X)
    recovered = scaler.inverse_transform(scaled)
    assert np.allclose(recovered, X)


@given(trade_logs())
@settings(max_examples=20)
def test_meta_label_bounds(df) -> None:
    prices = df["price"].to_numpy()
    spreads = df["spread"].to_numpy()
    tp = prices + spreads
    sl = prices - spreads
    _, _, _, meta = _compute_meta_labels(prices, tp, sl, hold_period=5)
    assert ((meta >= 0) & (meta <= 1)).all()
    assert np.isfinite(meta).all()


@given(feature_matrices(), model_parameters())
@settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
def test_manual_scaling_roundtrip(X: np.ndarray, params: dict) -> None:
    mean = np.array(params["feature_mean"])
    std = np.array(params["feature_std"])
    assume(X.shape[1] == len(mean))
    scaled = (X - mean) / std
    recovered = scaled * std + mean
    assert np.allclose(recovered, X)
