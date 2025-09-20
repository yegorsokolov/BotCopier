from pathlib import Path

import numpy as np
import pandas as pd

from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    configure_cache,
)


def test_cross_symbol_correlations(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "symbol": ["EURUSD", "USDCHF", "EURUSD", "USDCHF", "EURUSD", "USDCHF"],
            "price": [1.0, 1.2, 1.1, 1.25, 1.2, 1.3],
            "event_time": pd.date_range("2024-01-01", periods=6, freq="min"),
        }
    )
    feature_cols = ["price"]
    sg_path = Path("symbol_graph.json")
    config = configure_cache(FeatureConfig())
    df, feature_cols, _, _ = _extract_features(
        data.copy(),
        feature_cols,
        symbol_graph=sg_path,
        neighbor_corr_windows=[3],
        config=config,
    )
    corr_cols = ["corr_EURUSD_USDCHF_w3", "corr_USDCHF_EURUSD_w3"]
    for col in corr_cols:
        assert col in feature_cols
        assert df[col].between(-1, 1).all()
