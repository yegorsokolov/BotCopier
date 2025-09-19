import numpy as np
import sys
import types

import pandas as pd
import pytest

# stub minimal sklearn module to avoid heavy dependency when unavailable
try:
    import sklearn  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    sklearn = types.ModuleType("sklearn")
    sklearn.ensemble = types.ModuleType("sklearn.ensemble")
    sklearn.ensemble.IsolationForest = object
    sklearn.linear_model = types.ModuleType("sklearn.linear_model")
    sklearn.linear_model.LinearRegression = object
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.ensemble", sklearn.ensemble)
    sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)

# stub minimal scipy module when not installed
try:
    import scipy  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    scipy = types.ModuleType("scipy")
    scipy.signal = types.ModuleType("scipy.signal")
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", scipy.signal)

# stub gplearn
gplearn = types.ModuleType("gplearn")
gplearn.genetic = types.ModuleType("gplearn.genetic")
gplearn.genetic.SymbolicTransformer = object
sys.modules.setdefault("gplearn", gplearn)
sys.modules.setdefault("gplearn.genetic", gplearn.genetic)

# stub psutil
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

# stub joblib Memory
joblib = types.ModuleType("joblib")


class _DummyMemory:
    def __init__(self, *args, **kwargs):
        pass

    def cache(self, func):
        def wrapper(*a, **kw):
            return func(*a, **kw)

        wrapper.clear = lambda: None
        wrapper.check_call_in_cache = lambda *a, **kw: False
        return wrapper

    def clear(self):
        pass


joblib.Memory = _DummyMemory


class _DummyParallel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, iterable):
        return [func() for func in iterable]


def _dummy_delayed(func):
    def wrapper(*args, **kwargs):
        return lambda: func(*args, **kwargs)

    return wrapper


joblib.Parallel = _DummyParallel
joblib.delayed = _dummy_delayed
sys.modules.setdefault("joblib", joblib)

from botcopier.features.engineering import FeatureConfig, configure_cache
import botcopier.features.technical as technical
from botcopier.models.schema import FeatureMetadata

pywt = pytest.importorskip("pywt")


def _sample_wavelet_df(rows: int = 48) -> pd.DataFrame:
    time_index = pd.date_range("2023-01-01", periods=rows, freq="H")
    base = np.linspace(1.0, 1.5, rows)
    price = base + 0.01 * np.sin(np.linspace(0.0, 6.0, rows))
    volume = 100.0 + np.linspace(0.0, 10.0, rows) + 5.0 * np.cos(np.linspace(0.0, 3.0, rows))
    return pd.DataFrame(
        {
            "event_time": time_index,
            "symbol": ["EURUSD"] * rows,
            "price": price,
            "volume": volume,
        }
    )


def test_wavelet_packets_features_and_metadata():
    config = configure_cache(FeatureConfig(enabled_features={"wavelet_packets"}))
    try:
        df = _sample_wavelet_df()
        feature_names: list[str] = []
        out, feats, _, _ = technical._extract_features(
            df.copy(),
            feature_names,
            wavelet_windows=(16, 32),
            wavelet_stats=("mean", "energy"),
            config=config,
        )

        expected_columns = {
            "price_wp_w16_L1_mean",
            "price_wp_w16_L1_energy",
            "price_wp_w32_L1_mean",
            "price_wp_w32_L1_energy",
            "price_wp_w32_L2_mean",
            "price_wp_w32_L2_energy",
            "volume_wp_w16_L1_mean",
            "volume_wp_w16_L1_energy",
            "volume_wp_w32_L1_mean",
            "volume_wp_w32_L1_energy",
            "volume_wp_w32_L2_mean",
            "volume_wp_w32_L2_energy",
        }
        assert expected_columns.issubset(set(feats))

        for col in expected_columns:
            series = pd.to_numeric(out[col], errors="coerce")
            assert series.notna().all()
            assert np.isfinite(series).all()

        metadata_map = technical._FEATURE_METADATA
        for col in expected_columns:
            meta = metadata_map.get(col)
            assert meta is not None
            fm = FeatureMetadata(**meta)
            assert fm.original_column in {"price", "volume"}
            assert fm.transformations == ["wavelet_packets"]
            params = fm.parameters
            assert params["wavelet"] == pywt.Wavelet("db4").name
            assert params["window"] in {16, 32}
            assert params["level"] in {1, 2}
            assert params["statistic"] in {"mean", "energy"}

        # Cached invocation should reuse metadata without recomputation side effects.
        metadata_before = dict(metadata_map)
        out_cached, feats_cached, _, _ = technical._extract_features(
            out,
            list(feats),
            wavelet_windows=(16, 32),
            wavelet_stats=("mean", "energy"),
            config=config,
        )
        assert feats_cached == feats
        assert out_cached is out
        assert technical._FEATURE_METADATA == metadata_before
    finally:
        configure_cache(FeatureConfig())
