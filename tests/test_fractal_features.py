import sys
import types

import numpy as np
import pandas as pd

# Stub minimal sklearn dependency required by botcopier.features
sklearn = types.ModuleType("sklearn")
sklearn.ensemble = types.ModuleType("ensemble")
sklearn.ensemble.IsolationForest = object
sklearn.linear_model = types.ModuleType("linear_model")
sklearn.linear_model.LinearRegression = object
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.ensemble", sklearn.ensemble)
sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)
psutil = types.ModuleType("psutil")
sys.modules.setdefault("psutil", psutil)
joblib = types.ModuleType("joblib")
class _Cached:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def clear(self):
        pass
    def check_call_in_cache(self, *args, **kwargs):
        return False

class _Memory:
    def __init__(self, *args, **kwargs):
        pass
    def cache(self, func):
        return _Cached(func)

joblib.Memory = _Memory
sys.modules.setdefault("joblib", joblib)

from botcopier.features.technical import _extract_features_impl


def _random_walk(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def _trending_series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inc = np.zeros(n)
    phi = 0.8
    for i in range(1, n):
        inc[i] = phi * inc[i - 1] + rng.standard_normal()
    return np.cumsum(inc)


def test_random_walk_features():
    prices = _random_walk(200)
    df = pd.DataFrame({"symbol": ["EURUSD"] * len(prices), "price": prices})
    out, features, *_ = _extract_features_impl(df.copy(), [])
    assert "hurst" in features and "fractal_dim" in features
    h = out["hurst"].iloc[-1]
    f = out["fractal_dim"].iloc[-1]
    assert 0.3 < h < 0.7
    assert 1.3 < f < 1.7


def test_trending_series_features():
    prices = _trending_series(200)
    df = pd.DataFrame({"symbol": ["EURUSD"] * len(prices), "price": prices})
    out, *_ = _extract_features_impl(df.copy(), [])
    h = out["hurst"].iloc[-1]
    f = out["fractal_dim"].iloc[-1]
    assert h > 0.6
    assert f < 1.4
