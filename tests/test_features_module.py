from datetime import datetime

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import botcopier.features.technical as technical
from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    configure_cache,
)


def test_extract_features_basic():
    rows = [
        {
            "event_time": datetime(2024, 1, 1, 0, 0, 0),
            "action": "OPEN",
            "order_type": "0",
            "symbol": "EURUSD",
            "price": 1.1,
            "sl": 1.0,
            "tp": 1.2,
            "lots": 0.1,
            "profit": 1.0,
            "spread": 2.0,
        },
        {
            "event_time": datetime(2024, 1, 1, 1, 0, 0),
            "action": "OPEN",
            "order_type": "1",
            "symbol": "EURUSD",
            "price": 1.2,
            "sl": 1.25,
            "tp": 1.1,
            "lots": 0.1,
            "profit": -1.0,
            "spread": 2.0,
        },
    ]
    config = configure_cache(FeatureConfig())
    feats, labels, *_ = _extract_features(rows, config=config)
    assert len(feats) == 2
    assert labels.tolist() == [1, 0]
    assert "atr" in feats[0]
    assert "sl_dist_atr" in feats[0]
    assert "tp_dist_atr" in feats[0]


def test_mandatory_features_present():
    rows = [
        {
            "event_time": datetime(2024, 1, 1, 0, 0, 0),
            "action": "OPEN",
            "order_type": "0",
            "symbol": "EURUSD",
            "price": 1.1,
            "sl": 1.0,
            "tp": 1.2,
            "lots": 0.1,
            "profit": 1.0,
        }
    ]
    config = configure_cache(FeatureConfig())
    feats, *_ = _extract_features(rows, config=config)
    for key in ["book_bid_vol", "book_ask_vol", "book_imbalance", "equity", "margin_level", "atr"]:
        assert key in feats[0]


def test_news_embeddings_metadata():
    rows = [
        {
            "event_time": datetime(2024, 1, 1, 0, 0, 0),
            "symbol": "EURUSD",
            "price": 1.1,
            "profit": 0.1,
        },
        {
            "event_time": datetime(2024, 1, 1, 0, 1, 0),
            "symbol": "EURUSD",
            "price": 1.15,
            "profit": -0.2,
        },
    ]
    news = pd.DataFrame(
        [
            {"symbol": "EURUSD", "timestamp": "2024-01-01T00:00:30Z", "emb0": 0.5, "emb1": -0.25},
            {"symbol": "EURUSD", "timestamp": "2024-01-01T00:00:50Z", "emb0": 0.1, "emb1": 0.75},
        ]
    )
    config = configure_cache(FeatureConfig())
    df, feature_names, _, _ = _extract_features(
        rows,
        [],
        news_embeddings=news,
        news_embedding_window=2,
        news_embedding_horizon=120.0,
        config=config,
    )
    assert feature_names
    meta = technical._FEATURE_METADATA.get("__news_embeddings__")
    assert meta is not None
    assert meta["window"] == 2
    assert meta["dimension"] == 2
    sequences = np.array(meta["sequences"], dtype=float)
    assert sequences.shape == (len(df), 2, 2)
    # most recent embedding should be last row
    assert np.allclose(sequences[1, -1], np.array([0.1, 0.75]))
    technical._FEATURE_METADATA.clear()
