"""Data loading helpers for BotCopier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple
import hashlib

import numpy as np
import pandas as pd

from botcopier.features.augmentation import _augment_dataframe, _augment_dtw_dataframe

from ..scripts.data_validation import validate_logs


def _drop_duplicates_and_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Remove duplicate rows and outliers from ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    tuple[pd.DataFrame, int, int]
        The cleaned dataframe along with the counts of duplicate and
        outlier rows removed.
    """

    deduped = df.drop_duplicates()
    dup_count = len(df) - len(deduped)

    numeric = deduped.select_dtypes(include=[np.number])
    if numeric.empty:
        return deduped, dup_count, 0

    zscores = (numeric - numeric.mean()) / numeric.std(ddof=0)
    zscores = zscores.replace([np.inf, -np.inf], 0).fillna(0)
    outlier_mask = (np.abs(zscores) > 3).any(axis=1)
    out_count = int(outlier_mask.sum())

    cleaned = deduped.loc[~outlier_mask].copy()
    return cleaned, dup_count, out_count


def _compute_meta_labels(
    prices: np.ndarray, tp: np.ndarray, sl: np.ndarray, hold_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    mmap_threshold: int = 50 * 1024 * 1024,
    depth_file: Path | None = None,
) -> Tuple[Iterable[pd.DataFrame] | pd.DataFrame, list[str], dict[str, str]]:
    """Load trade logs from ``trades_raw.csv``.

    Parameters
    ----------
    mmap_threshold : int, optional
        If ``lite_mode`` is ``False`` and the input file exceeds this size in
        bytes, a memory-mapped/pyarrow-based reader is used to avoid loading the
        entire dataset into memory.

    Returns
    -------
    Tuple[Iterable[pd.DataFrame] | pd.DataFrame, list[str], dict[str, str]]
        Loaded logs, feature names, and a mapping of source file paths to
        SHA256 hashes.
    """
    file = data_dir if data_dir.is_file() else data_dir / "trades_raw.csv"
    if depth_file is None and not data_dir.is_file():
        cand = data_dir / "depth_raw.csv"
        depth_file = cand if cand.exists() else None
    cs = chunk_size or (50000 if lite_mode else None)
    use_mmap = (
        not lite_mode
        and cs is None
        and file.exists()
        and file.stat().st_size > mmap_threshold
    )
    data_hashes: dict[str, str] = {}
    if file.exists():
        data_hashes[str(file.resolve())] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    depth_df: pd.DataFrame | None = None
    if depth_file and depth_file.exists():
        import json

        depth_df = pd.read_csv(
            depth_file,
            converters={
                "bid_depth": lambda x: np.array(json.loads(x)) if isinstance(x, str) else x,
                "ask_depth": lambda x: np.array(json.loads(x)) if isinstance(x, str) else x,
            },
        )
        if "event_time" in depth_df.columns:
            depth_df["event_time"] = pd.to_datetime(depth_df["event_time"], errors="coerce")
        data_hashes[str(depth_file.resolve())] = hashlib.sha256(
            depth_file.read_bytes()
        ).hexdigest()

    def _process(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df.columns = [c.lower() for c in df.columns]
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        for col in df.columns:
            if col == "event_time":
                continue
            df[col] = pd.to_numeric(df[col], errors="ignore")

        if depth_df is not None:
            if "event_time" in df.columns and "event_time" in depth_df.columns:
                df = df.merge(depth_df, on="event_time", how="left")
            else:
                # Align by index if no event_time column
                join_cols = [c for c in depth_df.columns if c not in df.columns]
                df = pd.concat(
                    [
                        df.reset_index(drop=True),
                        depth_df[join_cols].iloc[: len(df)].reset_index(drop=True),
                    ],
                    axis=1,
                )

        df, dup_cnt, out_cnt = _drop_duplicates_and_outliers(df)
        logging.info("Removed %d duplicate rows and %d outlier rows", dup_cnt, out_cnt)

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
            dows = (
                pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
            )
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
            (
                c
                for c in ["net_profit", "profit", "price", "bid", "ask"]
                if c in df.columns
            ),
            None,
        )
        if price_col is not None:
            prices = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
            spread_src = (
                df["spread"]
                if "spread" in df.columns
                else pd.Series(0.0, index=df.index)
            )
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

        return df, feature_cols

    if kafka_brokers:
        import json

        from confluent_kafka import Consumer

        topic = data_dir.name if data_dir.is_file() else "trades_raw"
        consumer = Consumer(
            {
                "bootstrap.servers": kafka_brokers,
                "group.id": "botcopier",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([topic])

        def _kafka_iter():
            buffer: list[pd.DataFrame] = []
            count = 0
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    break
                data = json.loads(msg.value().decode("utf-8"))
                df_msg = pd.DataFrame([data])
                buffer.append(df_msg)
                count += len(df_msg)
                if not cs or count >= cs:
                    df_chunk = pd.concat(buffer, ignore_index=True)
                    df_chunk, fcols = _process(df_chunk)
                    buffer.clear()
                    count = 0
                    yield df_chunk, fcols
            if buffer:
                df_chunk = pd.concat(buffer, ignore_index=True)
                yield _process(df_chunk)
            consumer.close()

        kafka_stream = _kafka_iter()
        first_chunk, feature_cols = next(kafka_stream)

        def _iter():
            yield first_chunk
            for chunk, _ in kafka_stream:
                yield chunk

        return _iter(), feature_cols, data_hashes

    if flight_uri:
        from pyarrow import flight

        client = flight.FlightClient(flight_uri)
        descriptor = flight.FlightDescriptor.for_path(str(file).encode())
        info = client.get_flight_info(descriptor)
        reader = client.do_get(info.endpoints[0].ticket)

        first_batch = reader.read_next_batch()
        first_chunk, feature_cols = _process(first_batch.to_pandas())

        def _iter():
            yield first_chunk
            for batch in reader:
                df_chunk, _ = _process(batch.to_pandas())
                yield df_chunk

        return _iter(), feature_cols, data_hashes

    if use_mmap:
        try:  # Prefer pyarrow.dataset for lazy access when available
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(file), format="csv")
            batches = dataset.to_batches()
            first_batch = next(batches)
            first_chunk, feature_cols = _process(first_batch.to_pandas())

            def _iter():
                yield first_chunk
                for batch in batches:
                    df_chunk, _ = _process(batch.to_pandas())
                    yield df_chunk

            return _iter(), feature_cols, data_hashes
        except Exception:  # pragma: no cover - pyarrow.dataset missing
            reader = pd.read_csv(file, memory_map=True)
            df, feature_cols = _process(reader)
            return df, feature_cols, data_hashes

    reader = pd.read_csv(file, chunksize=cs, iterator=cs is not None)
    if cs:
        first_df = next(reader)
        first_df, feature_cols = _process(first_df)

        def _iter():
            yield first_df
            for chunk in reader:
                df_chunk, _ = _process(chunk)
                yield df_chunk

        return _iter(), feature_cols, data_hashes

    df = reader  # type: ignore[assignment]
    df, feature_cols = _process(df)
    return df, feature_cols, data_hashes
