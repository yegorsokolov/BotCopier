"""Data loading helpers for BotCopier."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import logging
import numpy as np
import pandas as pd

from ..scripts.data_validation import validate_logs
from botcopier.features.engineering import (
    _augment_dataframe,
    _augment_dtw_dataframe,
)


def _compute_meta_labels(prices: np.ndarray, tp: np.ndarray, sl: np.ndarray, hold_period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized computation of take-profit/stop-loss hit times and labels.

    Parameters
    ----------
    prices : np.ndarray
        Price series.
    tp : np.ndarray
        Take profit levels for each price.
    sl : np.ndarray
        Stop loss levels for each price.
    hold_period : int
        Maximum lookahead horizon.

    Returns
    -------
    horizon_idx : np.ndarray
        Index of the final lookahead point for each position.
    tp_time : np.ndarray
        Steps until take-profit is hit (``horizon_len + 1`` if never hit).
    sl_time : np.ndarray
        Steps until stop-loss is hit (``horizon_len + 1`` if never hit).
    meta : np.ndarray
        Meta label indicating whether take-profit was reached before
        stop-loss within the horizon.
    """

    n = len(prices)
    idx = np.arange(n)
    horizon_idx = np.minimum(idx + int(hold_period), n - 1)
    horizon_len = horizon_idx - idx

    offsets = np.arange(1, int(hold_period) + 1)
    future_idx = np.minimum(idx[:, None] + offsets, n - 1)
    future_prices = prices[future_idx]

    cummax = np.maximum.accumulate(future_prices, axis=1)
    cummin = np.minimum.accumulate(future_prices, axis=1)

    # searchsorted via counting elements below/above thresholds
    tp_hit = (cummax < tp[:, None]).sum(axis=1)
    sl_hit = (cummin > sl[:, None]).sum(axis=1)

    tp_time = np.where(tp_hit < horizon_len, tp_hit + 1, horizon_len + 1)
    sl_time = np.where(sl_hit < horizon_len, sl_hit + 1, horizon_len + 1)

    meta = (tp_time <= sl_time) & (tp_time <= horizon_len)

    return horizon_idx, tp_time, sl_time, meta.astype(float)


def _load_logs(
    data_dir: Path,
    *,
    lite_mode: bool | None = None,
    chunk_size: int | None = None,
    flight_uri: str | None = None,
    kafka_brokers: str | None = None,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    augment_ratio: float = 0.0,
    dtw_augment: bool = False,
) -> Tuple[Iterable[pd.DataFrame] | pd.DataFrame, list[str], list[str]]:
    """Load trade logs from ``trades_raw.csv``."""
    if kafka_brokers:
        raise NotImplementedError("kafka_brokers not supported")
    if flight_uri:
        raise NotImplementedError("flight_uri not supported")

    file = data_dir if data_dir.is_file() else data_dir / "trades_raw.csv"
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    for col in df.columns:
        if col == "event_time":
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    hours: pd.Series
    if "hour" in df.columns:
        hours = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    elif "event_time" in df.columns:
        hours = df["event_time"].dt.hour.fillna(0).astype(int)
        df["hour"] = hours
    else:
        hours = pd.Series(0, index=df.index, dtype=int)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    if "day_of_week" in df.columns:
        dows = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
        df.drop(columns=["day_of_week"], inplace=True)
    elif "event_time" in df.columns:
        dows = df["event_time"].dt.dayofweek.fillna(0).astype(int)
    else:
        dows = None
    if dows is not None:
        df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)

    if "event_time" in df.columns:
        months = df["event_time"].dt.month.fillna(1).astype(int)
        df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)
        doms = df["event_time"].dt.day.fillna(1).astype(int)
        df["dom_sin"] = np.sin(2 * np.pi * (doms - 1) / 31.0)
        df["dom_cos"] = np.cos(2 * np.pi * (doms - 1) / 31.0)

    optional_cols = [
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
    ]
    if dows is not None:
        optional_cols.extend(["dow_sin", "dow_cos"])
    feature_cols = [c for c in optional_cols if c in df.columns]

    validation_result = validate_logs(df)
    if not validation_result.get("success", False):
        logging.warning("Log validation failed: %s", validation_result)
        raise ValueError("log validation failed")
    logging.info(
        "Log validation succeeded: %s/%s expectations",
        validation_result.get("statistics", {}).get("successful_expectations", 0),
        validation_result.get("statistics", {}).get("evaluated_expectations", 0),
    )

    price_col = next(
        (c for c in ["net_profit", "profit", "price", "bid", "ask"] if c in df.columns),
        None,
    )
    if price_col is not None:
        prices = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
        spread_src = df["spread"] if "spread" in df.columns else pd.Series(0.0, index=df.index)
        spreads = pd.to_numeric(spread_src, errors="coerce").fillna(0.0)
        if not spreads.any():
            spreads = (prices.abs() * 0.001).fillna(0.0)
        tp = prices + take_profit_mult * spreads
        sl = prices - stop_loss_mult * spreads
        horizon_idx, tp_time, sl_time, meta = _compute_meta_labels(
            prices.to_numpy(), tp.to_numpy(), sl.to_numpy(), int(hold_period)
        )
        df["take_profit"] = tp
        df["stop_loss"] = sl
        df["horizon"] = horizon_idx
        df["tp_time"] = tp_time
        df["sl_time"] = sl_time
        df["meta_label"] = meta
        for horizon in (5, 20):
            label_name = f"label_h{horizon}"
            if label_name not in df.columns:
                pnl = prices.shift(-horizon) - prices
                df[label_name] = (pnl > 0).astype(float).fillna(0.0)

    if augment_ratio > 0:
        if dtw_augment:
            df = _augment_dtw_dataframe(df, augment_ratio)
        else:
            df = _augment_dataframe(df, augment_ratio)

    cs = chunk_size or (50000 if lite_mode else None)
    if cs:
        def _iter():
            for start in range(0, len(df), cs):
                yield df.iloc[start : start + cs]
        return _iter(), feature_cols, []

    return df, feature_cols, []
