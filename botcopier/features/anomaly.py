"""Anomaly detection and feature clipping utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


def _clip_train_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip features based on quantiles for robust training."""
    low = np.quantile(X, 0.01, axis=0)
    high = np.quantile(X, 0.99, axis=0)
    return _clip_apply(X, low, high), low, high


def _clip_apply(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(X, low, high)


def _score_anomalies(X: np.ndarray, params: dict | None) -> np.ndarray:
    if not params:
        return np.zeros(len(X))
    iso = IsolationForest(**params)
    return -iso.fit_predict(X)
