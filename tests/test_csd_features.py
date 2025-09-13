from pathlib import Path

import numpy as np
import pandas as pd

from botcopier.features import _extract_features
from botcopier.features.engineering import FeatureConfig, configure_cache


def test_csd_features_shapes_and_ranges(tmp_path):
    times = np.arange(32)
    data = []
    for sym, phase in [("EURUSD", 0.0), ("USDCHF", 0.5)]:
        prices = np.sin(2 * np.pi * 0.1 * times + phase)
        for t, p in zip(times, prices):
            data.append({"event_time": t, "symbol": sym, "price": p})
    df = pd.DataFrame(data)

    configure_cache(FeatureConfig(enabled_features={"csd"}))
    feats, cols, _, _ = _extract_features(
        df.copy(), [], symbol_graph=Path("symbol_graph.json")
    )
    configure_cache(FeatureConfig())

    assert {
        "csd_freq_USDCHF",
        "csd_coh_USDCHF",
        "csd_freq_EURUSD",
        "csd_coh_EURUSD",
    } <= set(cols)
    assert len(feats) == len(df)
    freq_cols = [c for c in feats.columns if c.startswith("csd_freq_")]
    coh_cols = [c for c in feats.columns if c.startswith("csd_coh_")]
    for c in freq_cols:
        series = feats[c].dropna()
        assert series.between(0.0, 0.5).all()
    for c in coh_cols:
        series = feats[c].dropna()
        assert series.between(0.0, 1.0).all()
