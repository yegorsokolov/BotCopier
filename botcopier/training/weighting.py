"""Sample weighting helpers for the training pipeline."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_time_decay_weights(
    event_times: np.ndarray | Sequence[object] | None, *, half_life_days: float
) -> np.ndarray:
    """Return exponential decay weights based on ``event_times``."""

    if event_times is None:
        return np.ones(0, dtype=float)

    dt_index = pd.to_datetime(pd.Index(event_times), errors="coerce", utc=True)
    try:
        dt_index = dt_index.tz_convert(None)
    except AttributeError:  # pragma: no cover - defensive for non-index inputs
        dt_index = pd.to_datetime(dt_index, errors="coerce")
    times = dt_index.to_numpy(dtype="datetime64[ns]")
    if times.size == 0:
        return np.ones(0, dtype=float)
    if half_life_days <= 0:
        return np.ones(times.size, dtype=float)

    mask = ~np.isnat(times)
    weights = np.ones(times.size, dtype=float)
    if not mask.any():
        return weights

    ref_time = times[mask].max()
    age_seconds = (ref_time - times[mask]).astype("timedelta64[s]").astype(float)
    age_days = age_seconds / (24 * 3600)
    decay = np.power(0.5, age_days / half_life_days)
    weights[mask] = decay
    if (~mask).any():
        fill_value = float(decay.min()) if decay.size else 1.0
        weights[~mask] = fill_value
    return weights


def compute_volatility_scaler(profits: np.ndarray, *, window: int = 50) -> np.ndarray:
    """Compute inverse volatility scaling factors for ``profits``."""

    if profits.size == 0:
        return np.ones(0, dtype=float)

    series = pd.Series(profits, dtype=float)
    vol = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
    scaler = 1.0 / (1.0 + vol.to_numpy(dtype=float))
    return np.clip(scaler, a_min=1e-6, a_max=None)


def compute_profit_weights(profits: np.ndarray) -> np.ndarray:
    """Return base sample weights derived from trade profits."""

    if profits.size == 0:
        return np.ones(0, dtype=float)

    weights = np.abs(np.asarray(profits, dtype=float))
    if not np.isfinite(weights).any():
        return np.ones_like(weights, dtype=float)
    median = np.median(weights[weights > 0]) if np.any(weights > 0) else 1.0
    if not np.isfinite(median) or median == 0:
        median = 1.0
    weights = weights / median
    weights = np.clip(weights, 0.0, 10.0)
    weights = np.nan_to_num(weights, nan=0.0, posinf=10.0, neginf=0.0)
    if not np.any(weights > 0):
        return np.ones_like(weights, dtype=float)
    return weights


def normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Return ``weights`` scaled to unit mean with NaNs replaced."""

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean()
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(arr, dtype=float)
    return arr / mean


def summarise_weights(weights: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for ``weights`` suitable for logging."""

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def build_sample_weights(
    profits: np.ndarray,
    event_times: np.ndarray | Sequence[object] | None,
    *,
    half_life_days: float,
    use_volatility: bool,
) -> np.ndarray:
    """Combine profit, time-decay and volatility based weights."""

    base = compute_profit_weights(profits)
    if event_times is not None and base.size:
        decay = compute_time_decay_weights(event_times, half_life_days=half_life_days)
        if decay.size == base.size:
            base = base * decay
    if use_volatility and base.size:
        scaler = compute_volatility_scaler(profits)
        if scaler.size == base.size:
            base = base * scaler
    return base


__all__ = [
    "build_sample_weights",
    "compute_profit_weights",
    "compute_time_decay_weights",
    "compute_volatility_scaler",
    "normalise_weights",
    "summarise_weights",
]
