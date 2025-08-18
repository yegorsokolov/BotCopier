import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


def _compute_decay_weights(event_times: np.ndarray, half_life_days: float) -> np.ndarray:
    ref_time = event_times.max()
    age_days = (
        (ref_time - event_times).astype("timedelta64[s]").astype(float) / (24 * 3600)
    )
    return 0.5 ** (age_days / half_life_days)


def test_time_series_split_sizes():
    n = 12
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(np.arange(n)))[-1]
    assert len(train_idx) == 10
    assert len(val_idx) == 2
    assert train_idx[-1] < val_idx[0]


def test_decay_weighting_changes_coefficients():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    times = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ],
        dtype="datetime64[s]",
    )
    weights = _compute_decay_weights(times, half_life_days=1.0)
    clf_plain = LogisticRegression().fit(X, y)
    clf_weighted = LogisticRegression().fit(X, y, sample_weight=weights)
    assert not np.allclose(clf_plain.coef_, clf_weighted.coef_)
