from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Memory

from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    configure_cache,
)
from botcopier.scripts.evaluation import _classification_metrics


def _sample_df():
    times = np.arange(64)
    data = []
    for sym, phase in [("EURUSD", 0.0), ("USDCHF", 0.5)]:
        prices = np.sin(2 * np.pi * 0.1 * times + phase)
        for t, p in zip(times, prices):
            data.append({"event_time": t, "symbol": sym, "price": p})
    return pd.DataFrame(data)


def test_feature_cache_speed(tmp_path):
    df = _sample_df()
    configure_cache(FeatureConfig(cache_dir=tmp_path, enabled_features={"csd"}))

    start = perf_counter()
    f1, c1, _, _ = _extract_features(
        df.copy(), [], symbol_graph=Path("symbol_graph.json")
    )
    t1 = perf_counter() - start

    start = perf_counter()
    f2, c2, _, _ = _extract_features(
        df.copy(), [], symbol_graph=Path("symbol_graph.json")
    )
    t2 = perf_counter() - start

    configure_cache(FeatureConfig())

    assert f1.equals(f2)
    assert c1 == c2
    assert t2 < t1


def test_classification_metrics_cache(tmp_path):
    mem = Memory(tmp_path, verbose=0)
    cached = mem.cache(_classification_metrics)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.random(1000)

    start = perf_counter()
    r1 = cached(y_true, y_pred, y_true * 0.0)
    t1 = perf_counter() - start

    start = perf_counter()
    r2 = cached(y_true, y_pred, y_true * 0.0)
    t2 = perf_counter() - start

    assert r1 == r2
    assert t2 < t1
