from datetime import datetime

from sklearn.feature_extraction import DictVectorizer

from scripts.features import _extract_features
from scripts.model_fitting import fit_logistic_regression
from scripts.evaluation import evaluate_model


def test_end_to_end_pipeline():
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
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feats)
    clf = fit_logistic_regression(X, labels)
    acc = evaluate_model(clf, X, labels)
    assert acc == 1.0
