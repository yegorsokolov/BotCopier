from datetime import datetime
import logging

import pandas as pd

from scripts.features import _extract_features as extract_rows_features
import botcopier.features.engineering as fe
from botcopier.features.engineering import configure_cache, clear_cache, FeatureConfig


def _synthetic_rows():
    return [
        {
            "event_time": datetime(2024, 1, 1, 0, 0, 0),
            "action": "OPEN",
            "order_type": "0",
            "symbol": "EURUSD",
            "price": 10.05,
            "label": 0,
            "open": 10.0,
            "high": 10.1,
            "low": 9.5,
            "close": 10.05,
            "sl": 9.0,
            "tp": 11.0,
            "lots": 0.1,
            "profit": 0.0,
        },
        {
            "event_time": datetime(2024, 1, 1, 1, 0, 0),
            "action": "OPEN",
            "order_type": "0",
            "symbol": "EURUSD",
            "price": 10.0,
            "label": 0,
            "open": 10.0,
            "high": 10.05,
            "low": 9.95,
            "close": 10.0,
            "sl": 9.0,
            "tp": 11.0,
            "lots": 0.1,
            "profit": 0.0,
        },
        {
            "event_time": datetime(2024, 1, 1, 2, 0, 0),
            "action": "OPEN",
            "order_type": "0",
            "symbol": "EURUSD",
            "price": 10.15,
            "label": 0,
            "open": 9.9,
            "high": 10.2,
            "low": 9.85,
            "close": 10.15,
            "sl": 9.0,
            "tp": 11.0,
            "lots": 0.1,
            "profit": 0.0,
        },
    ]


def test_pattern_detection_rows():
    feats, *_ = extract_rows_features(_synthetic_rows())
    assert feats[0]["pattern_hammer"] == 1.0
    assert feats[1]["pattern_doji"] == 1.0
    assert feats[2]["pattern_engulfing"] == 1.0


def test_pattern_detection_dataframe(tmp_path, caplog):
    cache_dir = tmp_path / "cache"
    configure_cache(FeatureConfig(cache_dir=cache_dir))
    clear_cache()
    rows = _synthetic_rows()
    df = pd.DataFrame(rows)
    feature_cols = ["price"]
    with caplog.at_level(logging.INFO):
        df, feature_cols, *_ = fe._extract_features(df, feature_cols)
        fe._extract_features(df, feature_cols)
    assert "cache hit for _extract_features" in caplog.text
    assert "pattern_hammer" in feature_cols
    assert "pattern_doji" in feature_cols
    assert "pattern_engulfing" in feature_cols
    assert df["pattern_hammer"].iloc[0] == 1.0
    assert df["pattern_doji"].iloc[1] == 1.0
    assert df["pattern_engulfing"].iloc[2] == 1.0
