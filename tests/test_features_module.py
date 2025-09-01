from datetime import datetime

from scripts.features import _extract_features


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
    feats, labels, *_ = _extract_features(rows)
    assert len(feats) == 2
    assert labels.tolist() == [1, 0]
