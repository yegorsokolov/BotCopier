#!/usr/bin/env python3
"""Train model from exported features.

The observer EA continuously exports trade logs as CSV files. This script
loads those logs, extracts a few simple features from each trade entry and
trains a very small predictive model. The resulting parameters along with
some training metadata are written to ``model.json`` so they can be consumed
by other helper scripts.
"""
import argparse
import json
import gzip
from datetime import datetime
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional
import sqlite3
import logging
import subprocess
import sys

import importlib.util

import pandas as pd

import numpy as np
import psutil
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


START_EVENT_ID = 0

try:  # Optional dependency for RL refinement
    import stable_baselines3  # type: ignore  # noqa: F401
    HAS_SB3 = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SB3 = False


def _has_sufficient_ram(min_gb: float = 4.0) -> bool:
    """Return True if the system has at least ``min_gb`` RAM."""
    try:
        return psutil.virtual_memory().total / (1024 ** 3) >= min_gb
    except Exception:  # pragma: no cover - psutil errors
        return False


def _has_sufficient_gpu(min_gb: float = 1.0) -> bool:
    """Return True if a CUDA GPU with ``min_gb`` memory is available."""
    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            return mem / (1024 ** 3) >= min_gb
    except Exception:
        pass
    return False


def detect_resources():
    """Detect available resources and installed ML libraries."""
    try:
        mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except Exception:  # pragma: no cover - psutil errors
        mem_gb = 0.0
        cores = 0

    lite_mode = mem_gb < 4 or (cores or 0) < 2

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    model_type = "logreg"
    if not lite_mode:
        for mt, module in [
            ("transformer", "transformers"),
            ("lstm", "torch"),
            ("catboost", "catboost"),
            ("lgbm", "lightgbm"),
            ("xgboost", "xgboost"),
            ("random_forest", "sklearn"),
        ]:
            if has(module):
                model_type = mt
                break

    use_optuna = (
        not lite_mode
        and has("optuna")
        and mem_gb >= 8
        and (cores or 0) >= 4
    )
    optuna_trials = 20 if use_optuna else 0
    return {
        "lite_mode": lite_mode,
        "model_type": model_type,
        "optuna_trials": optuna_trials,
    }


def _export_onnx(clf, feature_names: List[str], out_dir: Path) -> None:
    """Export ``clf`` to ONNX format if dependencies are available."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_path = out_dir / "model.onnx"
        initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
        model_onnx = convert_sklearn(clf, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        print(f"ONNX model written to {onnx_path}")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"ONNX conversion skipped: {exc}")


def _sma(values, window):
    """Simple moving average for the last ``window`` values."""
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(sum(values[-w:]) / w)


def _atr(values, period):
    """Average true range over ``period`` price changes."""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values[-(period + 1) :]))
    if len(diffs) == 0:
        return 0.0
    w = min(period, len(diffs))
    return float(diffs[-w:].mean())


def _bollinger(values, window, dev=2.0):
    """Return Bollinger Bands for the last ``window`` values."""
    if not values:
        return 0.0, 0.0, 0.0
    w = min(window, len(values))
    arr = np.array(values[-w:])
    sma = arr.mean()
    std = arr.std(ddof=0)
    upper = sma + dev * std
    lower = sma - dev * std
    return float(upper), float(sma), float(lower)


def _safe_float(val, default=0.0):
    """Convert ``val`` to float, treating ``None``/NaN as ``default``."""
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def _rsi(values, period):
    """Very small RSI implementation on ``values``."""
    if len(values) < 2:
        return 50.0
    deltas = np.diff(values[-(period + 1) :])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _macd_update(state, price, short=12, long=26, signal=9):
    """Update MACD EMA state with ``price`` and return (macd, signal)."""
    alpha_short = 2 / (short + 1)
    alpha_long = 2 / (long + 1)
    alpha_signal = 2 / (signal + 1)

    ema_short = state.get("ema_short")
    ema_long = state.get("ema_long")
    ema_signal = state.get("ema_signal")

    ema_short = price if ema_short is None else alpha_short * price + (1 - alpha_short) * ema_short
    ema_long = price if ema_long is None else alpha_long * price + (1 - alpha_long) * ema_long
    macd = ema_short - ema_long
    ema_signal = macd if ema_signal is None else alpha_signal * macd + (1 - alpha_signal) * ema_signal

    state["ema_short"] = ema_short
    state["ema_long"] = ema_long
    state["ema_signal"] = ema_signal

    return macd, ema_signal


def _stochastic_update(state, price, k_period=14, d_period=3):
    """Update and return stochastic %K and %D values."""
    prices = state.setdefault("prices", [])
    prices.append(price)
    if len(prices) > k_period:
        del prices[0]
    low = min(prices)
    high = max(prices)
    if high == low:
        k = 0.0
    else:
        k = (price - low) / (high - low) * 100.0
    k_history = state.setdefault("k_values", [])
    k_history.append(k)
    if len(k_history) > d_period:
        del k_history[0]
    d = float(sum(k_history) / len(k_history))
    return float(k), d


def _adx_update(state, price, period=14):
    """Update ADX state with ``price`` and return current ADX."""
    prev = state.get("prev_price")
    state["prev_price"] = price
    if prev is None:
        return 0.0

    up_move = price - prev if price > prev else 0.0
    down_move = prev - price if price < prev else 0.0
    tr = abs(price - prev)

    plus_dm = state.setdefault("plus_dm", [])
    minus_dm = state.setdefault("minus_dm", [])
    tr_list = state.setdefault("tr", [])
    dx_list = state.setdefault("dx", [])

    plus_dm.append(up_move)
    minus_dm.append(down_move)
    tr_list.append(tr)
    if len(plus_dm) > period:
        del plus_dm[0]
    if len(minus_dm) > period:
        del minus_dm[0]
    if len(tr_list) > period:
        del tr_list[0]

    atr = sum(tr_list) / len(tr_list)
    if atr == 0:
        di_plus = di_minus = 0.0
    else:
        di_plus = 100.0 * (sum(plus_dm) / len(plus_dm)) / atr
        di_minus = 100.0 * (sum(minus_dm) / len(minus_dm)) / atr

    denom = di_plus + di_minus
    dx = 0.0 if denom == 0 else 100.0 * abs(di_plus - di_minus) / denom
    dx_list.append(dx)
    if len(dx_list) > period:
        del dx_list[0]
    adx = sum(dx_list) / len(dx_list)
    return float(adx)


def _rolling_corr(a, b, window=5):
    """Return correlation of the last ``window`` points of ``a`` and ``b``."""
    if not a or not b:
        return 0.0
    w = min(len(a), len(b), window)
    if w < 2:
        return 0.0
    arr1 = np.array(a[-w:], dtype=float)
    arr2 = np.array(b[-w:], dtype=float)
    if arr1.std(ddof=0) == 0 or arr2.std(ddof=0) == 0:
        return 0.0
    return float(np.corrcoef(arr1, arr2)[0, 1])


def _load_logs_db(db_file: Path) -> pd.DataFrame:
    """Load log rows from a SQLite database."""

    conn = sqlite3.connect(db_file)
    try:
        query = "SELECT * FROM logs"
        params: tuple = ()
        if START_EVENT_ID > 0:
            query += " WHERE CAST(event_id AS INTEGER) > ?"
            params = (START_EVENT_ID,)
        df_logs = pd.read_sql_query(query, conn, params=params, parse_dates=["event_time", "open_time"])
    finally:
        conn.close()

    df_logs.columns = [c.lower() for c in df_logs.columns]

    if "open_time" in df_logs.columns:
        df_logs["trade_duration"] = (
            pd.to_datetime(df_logs["event_time"]) - pd.to_datetime(df_logs["open_time"])
        ).dt.total_seconds().fillna(0)
    for col in ["book_bid_vol", "book_ask_vol", "book_imbalance"]:
        if col not in df_logs.columns:
            df_logs[col] = 0.0
        else:
            df_logs[col] = pd.to_numeric(df_logs[col], errors="coerce").fillna(0.0)
    if "is_anomaly" not in df_logs.columns:
        df_logs["is_anomaly"] = 0
    else:
        df_logs["is_anomaly"] = (
            pd.to_numeric(df_logs["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    if "action" in df_logs.columns:
        df_logs["action"] = df_logs["action"].fillna("").str.upper()
        df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    invalid_rows = pd.DataFrame(columns=df_logs.columns)
    if "event_id" in df_logs.columns:
        dup_mask = df_logs.duplicated(subset="event_id", keep="first")
        if dup_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[dup_mask]])
            logging.warning("Dropping %s duplicate event_id rows", dup_mask.sum())
        df_logs = df_logs[~dup_mask]

    if set(["ticket", "action"]).issubset(df_logs.columns):
        crit_mask = (
            df_logs["ticket"].isna()
            | (df_logs["ticket"].astype(str) == "")
            | df_logs["action"].isna()
            | (df_logs["action"].astype(str) == "")
        )
        if crit_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[crit_mask]])
            logging.warning("Dropping %s rows with missing ticket/action", crit_mask.sum())
        df_logs = df_logs[~crit_mask]

    if "lots" in df_logs.columns:
        df_logs["lots"] = pd.to_numeric(df_logs["lots"], errors="coerce")
    if "price" in df_logs.columns:
        df_logs["price"] = pd.to_numeric(df_logs["price"], errors="coerce")
    unreal_mask = pd.Series(False, index=df_logs.index)
    if "lots" in df_logs.columns:
        unreal_mask |= df_logs["lots"] < 0
    if "price" in df_logs.columns:
        unreal_mask |= df_logs["price"].isna()
    if unreal_mask.any():
        invalid_rows = pd.concat([invalid_rows, df_logs[unreal_mask]])
        logging.warning("Dropping %s rows with negative lots or NaN price", unreal_mask.sum())
    df_logs = df_logs[~unreal_mask]

    if not invalid_rows.empty:
        invalid_file = db_file.with_name("invalid_rows.csv")
        try:
            invalid_rows.to_csv(invalid_file, index=False)
        except Exception:  # pragma: no cover - disk issues
            pass

    return df_logs


def _load_logs(
    data_dir: Path,
    *,
    lite_mode: bool = False,
    chunk_size: int = 50000,
) -> tuple[pd.DataFrame | Iterable[pd.DataFrame], list[str], list[str]]:
    """Load log rows from ``data_dir``.

    ``MODIFY`` entries are retained alongside ``OPEN`` and ``CLOSE``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files or a SQLite ``.db`` file.

    Returns
    -------
    tuple[pandas.DataFrame, list[str], list[str]]
        Parsed rows as a DataFrame along with commit hashes and checksums
        collected from accompanying manifest files.
    """

    if data_dir.suffix == ".db":
        df = _load_logs_db(data_dir)
        return df, [], []

    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "sl_dist",
        "tp_dist",
        "sl_hit_dist",
        "tp_hit_dist",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
        "open_time",
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "is_anomaly",
    ]

    data_commits: list[str] = []
    data_checksums: list[str] = []

    metrics_file = data_dir / "metrics.csv"
    df_metrics = None
    key_col: str | None = None
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file, sep=";")
        df_metrics.columns = [c.lower() for c in df_metrics.columns]
        if "magic" in df_metrics.columns:
            key_col = "magic"
        elif "model_id" in df_metrics.columns:
            key_col = "model_id"

    invalid_file = data_dir / "invalid_rows.csv"

    def iter_chunks():
        seen_ids: set[str] = set()
        invalid_rows: list[pd.DataFrame] = []
        for log_file in sorted(data_dir.glob("trades_*.csv")):
            reader = pd.read_csv(
                log_file,
                sep=";",
                header=0,
                dtype=str,
                chunksize=chunk_size,
                engine="python",
            )
            manifest_file = log_file.with_suffix(".manifest.json")
            if manifest_file.exists():
                try:
                    with open(manifest_file) as mf:
                        meta = json.load(mf)
                    commit = meta.get("commit")
                    checksum = meta.get("checksum")
                    if commit:
                        data_commits.append(str(commit))
                    if checksum:
                        data_checksums.append(str(checksum))
                except Exception:
                    pass
            for chunk in reader:
                chunk = chunk.reindex(columns=fields)
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk["event_time"] = pd.to_datetime(chunk.get("event_time"), errors="coerce")
                if "open_time" in chunk.columns:
                    chunk["open_time"] = pd.to_datetime(chunk.get("open_time"), errors="coerce")
                    chunk["trade_duration"] = (
                        chunk["event_time"] - chunk["open_time"]
                    ).dt.total_seconds().fillna(0)
                else:
                    chunk["trade_duration"] = 0.0
                for col in ["book_bid_vol", "book_ask_vol", "book_imbalance"]:
                    chunk[col] = pd.to_numeric(chunk.get(col, 0.0), errors="coerce").fillna(0.0)
                chunk["is_anomaly"] = pd.to_numeric(chunk.get("is_anomaly", 0), errors="coerce").fillna(0)
                valid_actions = {"OPEN", "CLOSE", "MODIFY"}
                chunk["action"] = chunk["action"].fillna("").str.upper()
                chunk = chunk[(chunk["action"] == "") | chunk["action"].isin(valid_actions)]
                invalid = pd.DataFrame(columns=chunk.columns)
                if "event_id" in chunk.columns:
                    dup_mask = (
                        chunk["event_id"].isin(seen_ids)
                        | chunk.duplicated(subset="event_id", keep="first")
                    )
                    if dup_mask.any():
                        invalid = pd.concat([invalid, chunk[dup_mask]])
                        logging.warning(
                            "Dropping %s duplicate event_id rows", dup_mask.sum()
                        )
                    seen_ids.update(chunk.loc[~dup_mask, "event_id"].tolist())
                    chunk = chunk[~dup_mask]
                if {"ticket", "action"}.issubset(chunk.columns):
                    crit_mask = (
                        chunk["ticket"].isna()
                        | (chunk["ticket"].astype(str) == "")
                        | chunk["action"].isna()
                        | (chunk["action"].astype(str) == "")
                    )
                    if crit_mask.any():
                        invalid = pd.concat([invalid, chunk[crit_mask]])
                        logging.warning(
                            "Dropping %s rows with missing ticket/action", crit_mask.sum()
                        )
                    chunk = chunk[~crit_mask]
                if "lots" in chunk.columns:
                    chunk["lots"] = pd.to_numeric(chunk["lots"], errors="coerce")
                if "price" in chunk.columns:
                    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                unreal_mask = pd.Series(False, index=chunk.index)
                if "lots" in chunk.columns:
                    unreal_mask |= chunk["lots"] < 0
                if "price" in chunk.columns:
                    unreal_mask |= chunk["price"].isna()
                if unreal_mask.any():
                    invalid = pd.concat([invalid, chunk[unreal_mask]])
                    logging.warning(
                        "Dropping %s rows with negative lots or NaN price", unreal_mask.sum()
                    )
                chunk = chunk[~unreal_mask]
                if df_metrics is not None and key_col is not None and "magic" in chunk.columns:
                    chunk["magic"] = (
                        pd.to_numeric(chunk["magic"], errors="coerce").fillna(0).astype(int)
                    )
                    if key_col == "magic":
                        chunk = chunk.merge(df_metrics, how="left", on="magic")
                    else:
                        chunk = chunk.merge(
                            df_metrics, how="left", left_on="magic", right_on="model_id"
                        )
                        chunk = chunk.drop(columns=["model_id"])
                if not invalid.empty:
                    invalid_rows.append(invalid)
                yield chunk
        if invalid_rows:
            try:
                pd.concat(invalid_rows, ignore_index=True).to_csv(invalid_file, index=False)
            except Exception:  # pragma: no cover - disk issues
                pass

    if lite_mode:
        return iter_chunks(), data_commits, data_checksums
    dfs = list(iter_chunks())
    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=[c.lower() for c in fields])
    return df_logs, data_commits, data_checksums


def _load_calendar(file: Path) -> list[tuple[datetime, float]]:
    """Load calendar events from a CSV file.

    The file is expected to have at least two columns: ``time`` and
    ``impact``. ``time`` should be parseable by ``pandas.to_datetime``.

    Parameters
    ----------
    file : Path
        CSV file containing calendar events.

    Returns
    -------
    list[tuple[datetime, float]]
        Sorted list of ``(event_time, impact)`` tuples.
    """

    if not file.exists():
        return []
    df = pd.read_csv(file)
    events: list[tuple[datetime, float]] = []
    for _, row in df.iterrows():
        t = pd.to_datetime(row.get("time"), utc=False, errors="coerce")
        if pd.isna(t):
            continue
        impact = float(row.get("impact", 0.0) or 0.0)
        events.append((t.to_pydatetime(), impact))
    events.sort(key=lambda x: x[0])
    return events


def _read_last_event_id(out_dir: Path) -> int:
    """Read ``last_event_id`` from an existing model file in ``out_dir``."""
    json_path = out_dir / "model.json"
    gz_path = out_dir / "model.json.gz"
    model_file: Path | None = None
    open_func = open
    if gz_path.exists():
        model_file = gz_path
        open_func = gzip.open
    elif json_path.exists():
        model_file = json_path
    if model_file is None:
        return 0
    try:
        with open_func(model_file, "rt") as f:
            data = json.load(f)
        return int(data.get("last_event_id", 0))
    except Exception:
        return 0


def _extract_features(
    rows,
    use_sma=False,
    sma_window=5,
    use_rsi=False,
    rsi_period=14,
    use_macd=False,
    use_atr=False,
    atr_period=14,
    use_bollinger=False,
    boll_window=20,
    use_stochastic=False,
    use_adx=False,
    use_volume=False,
    volatility=None,
    higher_timeframes=None,
    *,
    corr_pairs=None,
    extra_price_series=None,
    corr_window: int = 5,
    encoder: dict | None = None,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    perf_budget: float | None = None,
):
    feature_dicts = []
    labels = []
    sl_targets = []
    tp_targets = []
    prices = []
    hours = []
    macd_state = {}
    higher_timeframes = [str(tf).upper() for tf in (higher_timeframes or [])]
    tf_map = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }
    tf_secs = {tf: tf_map.get(tf, 60) * 60 for tf in higher_timeframes}
    tf_prices = {tf: [] for tf in higher_timeframes}
    tf_macd_state = {tf: {} for tf in higher_timeframes}
    tf_macd = {tf: 0.0 for tf in higher_timeframes}
    tf_macd_sig = {tf: 0.0 for tf in higher_timeframes}
    tf_last_bin = {tf: None for tf in higher_timeframes}
    tf_prev_price = {tf: None for tf in higher_timeframes}
    stoch_state = {}
    adx_state = {}
    extra_series = extra_price_series or {}
    price_map = {sym: [] for sym in extra_series.keys()}
    enc_window = int(encoder.get("window")) if encoder else 0
    enc_weights = (
        np.array(encoder.get("weights", []), dtype=float) if encoder else np.empty((0, 0))
    )
    enc_centers = (
        np.array(encoder.get("centers", []), dtype=float) if encoder else np.empty((0, 0))
    )
    calendar_events = calendar_events or []
    row_idx = 0

    start_time = time.perf_counter()
    psutil.cpu_percent(interval=None)
    heavy_order = ["multi_tf", "use_adx", "use_stochastic", "use_bollinger", "use_atr"]

    for r in rows:
        if r.get("action", "").upper() != "OPEN":
            continue
        if str(r.get("is_anomaly", "0")).lower() in ("1", "true", "yes"):
            continue

        t = r["event_time"]
        if not isinstance(t, datetime):
            parsed = None
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
                try:
                    parsed = datetime.strptime(str(t), fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                continue
            t = parsed

        order_type = int(float(r.get("order_type", 0)))
        label = 1 if order_type == 0 else 0  # buy=1, sell=0

        price = _safe_float(r.get("price", 0))
        sl = _safe_float(r.get("sl", 0))
        tp = _safe_float(r.get("tp", 0))
        lots = _safe_float(r.get("lots", 0))
        profit = _safe_float(r.get("profit", 0))

        for tf in higher_timeframes:
            tf_bin = int(t.timestamp() // tf_secs[tf])
            if tf_last_bin[tf] is None:
                tf_last_bin[tf] = tf_bin
            elif tf_bin != tf_last_bin[tf]:
                if tf_prev_price[tf] is not None:
                    tf_prices[tf].append(tf_prev_price[tf])
                    if use_macd:
                        tf_macd[tf], tf_macd_sig[tf] = _macd_update(
                            tf_macd_state[tf], tf_prev_price[tf]
                        )
                tf_last_bin[tf] = tf_bin
            tf_prev_price[tf] = price

        symbol = r.get("symbol", "")
        sym_prices = price_map.setdefault(symbol, [])

        spread = _safe_float(r.get("spread", 0))
        slippage = _safe_float(r.get("slippage", 0))
        account_equity = _safe_float(r.get("equity", 0))
        margin_level = _safe_float(r.get("margin_level", 0))

        hour_sin = math.sin(2 * math.pi * t.hour / 24)
        hour_cos = math.cos(2 * math.pi * t.hour / 24)
        dow_sin = math.sin(2 * math.pi * t.weekday() / 7)
        dow_cos = math.cos(2 * math.pi * t.weekday() / 7)

        sl_dist = _safe_float(r.get("sl_dist", sl - price))
        tp_dist = _safe_float(r.get("tp_dist", tp - price))
        sl_hit = _safe_float(r.get("sl_hit_dist", 0.0))
        tp_hit = _safe_float(r.get("tp_hit_dist", 0.0))

        feat = {
            "symbol": symbol,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "lots": lots,
            "profit": profit,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "sl_hit_dist": sl_hit,
            "tp_hit_dist": tp_hit,
            "spread": spread,
            "slippage": slippage,
            "equity": account_equity,
            "margin_level": margin_level,
            "book_bid_vol": float(r.get("book_bid_vol", 0) or 0),
            "book_ask_vol": float(r.get("book_ask_vol", 0) or 0),
            "book_imbalance": float(r.get("book_imbalance", 0) or 0),
        }

        if calendar_events is not None:
            flag = 0.0
            impact_val = 0.0
            for ev_time, ev_imp in calendar_events:
                if abs((t - ev_time).total_seconds()) <= event_window * 60.0:
                    flag = 1.0
                    if ev_imp > impact_val:
                        impact_val = ev_imp
            feat["event_flag"] = flag
            feat["event_impact"] = impact_val

        if use_volume:
            feat["volume"] = float(r.get("volume", 0) or 0)

        if volatility is not None:
            key = t.strftime("%Y-%m-%d %H")
            vol = volatility.get(key)
            if vol is None:
                key = t.strftime("%Y-%m-%d")
                vol = volatility.get(key, 0.0)
            feat["volatility"] = float(vol)

        if use_sma:
            feat["sma"] = _sma(prices, sma_window)

        if use_rsi:
            feat["rsi"] = _rsi(prices, rsi_period)

        if use_macd:
            macd, signal = _macd_update(macd_state, price)
            feat["macd"] = macd
            feat["macd_signal"] = signal

        if use_atr:
            feat["atr"] = _atr(prices, atr_period)

        if use_bollinger:
            upper, mid, lower = _bollinger(prices, boll_window)
            feat["bollinger_upper"] = upper
            feat["bollinger_middle"] = mid
            feat["bollinger_lower"] = lower

        if use_stochastic:
            k, d_val = _stochastic_update(stoch_state, price)
            feat["stochastic_k"] = k
            feat["stochastic_d"] = d_val

        if use_adx:
            feat["adx"] = _adx_update(adx_state, price)

        for tf in higher_timeframes:
            prices_tf = tf_prices.get(tf, [])
            if use_sma:
                feat[f"sma_{tf}"] = _sma(prices_tf, sma_window)
            if use_rsi:
                feat[f"rsi_{tf}"] = _rsi(prices_tf, rsi_period)
            if use_macd:
                feat[f"macd_{tf}"] = tf_macd.get(tf, 0.0)
                feat[f"macd_signal_{tf}"] = tf_macd_sig.get(tf, 0.0)

        if corr_pairs:
            for s1, s2 in corr_pairs:
                p1 = price_map.get(s1, [])
                p2 = price_map.get(s2, [])
                corr = _rolling_corr(p1, p2, corr_window)
                ratio = 0.0
                if p1 and p2 and p2[-1] != 0:
                    ratio = p1[-1] / p2[-1]
                feat[f"corr_{s1}_{s2}"] = corr
                feat[f"ratio_{s1}_{s2}"] = ratio

        if enc_window > 0 and enc_weights.size > 0:
            seq = (prices + [price])[-(enc_window + 1) :]
            if len(seq) < enc_window + 1:
                seq = [seq[0]] * (enc_window + 1 - len(seq)) + seq
            deltas = np.diff(seq)
            vals = deltas.dot(enc_weights)
            for i, v in enumerate(vals):
                feat[f"ae{i}"] = float(v)
            if enc_centers.size > 0:
                d = ((enc_centers - vals) ** 2).sum(axis=1)
                feat["regime"] = float(int(np.argmin(d)))

        prices.append(price)
        sym_prices.append(price)
        for sym, series in extra_series.items():
            if sym == symbol:
                continue
            if row_idx < len(series):
                price_map.setdefault(sym, []).append(float(series[row_idx]))

        feature_dicts.append(feat)
        labels.append(label)
        sl_targets.append(sl_dist)
        tp_targets.append(tp_dist)
        hours.append(t.hour)
        row_idx += 1

        if perf_budget is not None:
            elapsed = time.perf_counter() - start_time
            load = psutil.cpu_percent(interval=None)
            while heavy_order and (
                elapsed > perf_budget * row_idx or load > 90.0
            ):
                feat_name = heavy_order.pop(0)
                if feat_name == "use_atr":
                    use_atr = False
                elif feat_name == "use_bollinger":
                    use_bollinger = False
                elif feat_name == "use_stochastic":
                    use_stochastic = False
                elif feat_name == "use_adx":
                    use_adx = False
                elif feat_name == "multi_tf":
                    higher_timeframes = []
                    tf_prices.clear()
                    tf_macd_state.clear()
                    tf_macd.clear()
                    tf_macd_sig.clear()
                    tf_last_bin.clear()
                    tf_prev_price.clear()
                logging.info("Disabling %s due to performance budget", feat_name)
                elapsed = time.perf_counter() - start_time
                load = psutil.cpu_percent(interval=None)

    enabled_feats = []
    if use_sma:
        enabled_feats.append("sma")
    if use_rsi:
        enabled_feats.append("rsi")
    if use_macd:
        enabled_feats.append("macd")
    if use_atr:
        enabled_feats.append("atr")
    if use_bollinger:
        enabled_feats.append("bollinger")
    if use_stochastic:
        enabled_feats.append("stochastic")
    if use_adx:
        enabled_feats.append("adx")
    if higher_timeframes:
        enabled_feats.extend(f"tf_{tf}" for tf in higher_timeframes)
    logging.info("Enabled features: %s", sorted(enabled_feats))

    return (
        feature_dicts,
        np.array(labels),
        np.array(sl_targets),
        np.array(tp_targets),
        np.array(hours, dtype=int),
    )


def _best_threshold(y_true, probas):
    """Return probability threshold with best F1 score."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probas >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def _train_lite_mode(
    data_dir: Path,
    out_dir: Path,
    *,
    use_sma: bool = False,
    sma_window: int = 5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    use_macd: bool = False,
    use_atr: bool = False,
    atr_period: int = 14,
    use_bollinger: bool = False,
    boll_window: int = 20,
    use_stochastic: bool = False,
    use_adx: bool = False,
    use_volume: bool = False,
    volatility_series=None,
    corr_pairs=None,
    corr_window: int = 5,
    extra_price_series=None,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    encoder_file: Path | None = None,
    chunk_size: int = 50000,
    compress_model: bool = False,
    regime_model: dict | None = None,
) -> None:
    """Stream features and train an SGD classifier incrementally."""

    ae_info = None
    rows_iter, data_commits, data_checksums = _load_logs(
        data_dir, lite_mode=True, chunk_size=chunk_size
    )
    last_event_id = 0
    encoder = None
    if encoder_file is not None and encoder_file.exists():
        with open(encoder_file) as f:
            encoder = json.load(f)

    vec = DictVectorizer(sparse=False)
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss")
    if regime_model is not None:
        vec_reg = DictVectorizer(sparse=False)
        vec_reg.fit([{n: 0.0} for n in regime_model.get("feature_names", [])])
        reg_mean = np.array(regime_model.get("mean", []), dtype=float)
        reg_std = np.array(regime_model.get("std", []), dtype=float)
        reg_std[reg_std == 0] = 1.0
        reg_centers = np.array(regime_model.get("centers", []), dtype=float)
    else:
        vec_reg = None
        reg_mean = reg_std = reg_centers = None
    first = True
    sample_count = 0

    for chunk in rows_iter:
        if "event_id" in chunk.columns:
            max_id = pd.to_numeric(chunk["event_id"], errors="coerce").max()
            if not pd.isna(max_id):
                last_event_id = max(last_event_id, int(max_id))
        (
            f_chunk,
            l_chunk,
            _,
            _,
            _,
        ) = _extract_features(
            chunk.to_dict("records"),
            use_sma=use_sma,
            sma_window=sma_window,
            use_rsi=use_rsi,
            rsi_period=rsi_period,
            use_macd=use_macd,
            use_atr=use_atr,
            atr_period=atr_period,
            use_bollinger=use_bollinger,
            boll_window=boll_window,
            use_stochastic=use_stochastic,
            use_adx=use_adx,
            use_volume=use_volume,
            volatility=volatility_series,
            higher_timeframes=None,
            corr_pairs=corr_pairs,
            corr_window=corr_window,
            extra_price_series=extra_price_series,
            encoder=encoder,
            calendar_events=calendar_events,
            event_window=event_window,
        )
        if not f_chunk:
            continue
        if vec_reg is not None and f_chunk:
            Xr = vec_reg.transform(f_chunk)
            Xr = (Xr - reg_mean) / reg_std
            dists = ((Xr[:, None, :] - reg_centers[None, :, :]) ** 2).sum(axis=2)
            rids = dists.argmin(axis=1)
            for i, r in enumerate(rids):
                f_chunk[i][f"regime_{int(r)}"] = 1.0
        sample_count += len(l_chunk)
        if first:
            X = vec.fit_transform(f_chunk)
            scaler.partial_fit(X)
            X = scaler.transform(X)
            clf.partial_fit(X, l_chunk, classes=np.array([0, 1]))
            first = False
        else:
            X = vec.transform(f_chunk)
            scaler.partial_fit(X)
            X = scaler.transform(X)
            clf.partial_fit(X, l_chunk)

    if first:
        raise ValueError(f"No training data found in {data_dir}")

    feature_names = vec.get_feature_names_out().tolist()
    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "weighted": False,
        "train_accuracy": float("nan"),
        "val_accuracy": float("nan"),
        "threshold": 0.5,
        "accuracy": float("nan"),
        "num_samples": int(sample_count),
        "feature_importance": {},
        "mean": scaler.mean_.astype(np.float32).tolist(),
        "std": scaler.scale_.astype(np.float32).tolist(),
        "coefficients": clf.coef_[0].astype(np.float32).tolist(),
        "intercept": float(clf.intercept_[0]),
        "classes": [int(c) for c in clf.classes_],
        "last_event_id": int(last_event_id),
    }
    if ae_info:
        model["autoencoder"] = ae_info
    if regime_model is not None and reg_centers is not None:
        model["regime_centers"] = reg_centers.astype(float).tolist()
        model["regime_feature_names"] = regime_model.get("feature_names", [])
    if data_commits:
        model["data_commit"] = ",".join(sorted(set(data_commits)))
    if data_checksums:
        model["data_checksum"] = ",".join(sorted(set(data_checksums)))
    if calendar_events:
        model["calendar_events"] = [
            [dt.isoformat(), float(imp)] for dt, imp in calendar_events
        ]
        model["event_window"] = float(event_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f)
    print(f"Model written to {model_path}")
    _export_onnx(clf, feature_names, out_dir)


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    use_sma: bool = False,
    sma_window: int = 5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    use_macd: bool = False,
    use_atr: bool = False,
    atr_period: int = 14,
    use_bollinger: bool = False,
    boll_window: int = 20,
    use_stochastic: bool = False,
    use_adx: bool = False,
    use_volume: bool = False,
    volatility_series=None,
    higher_timeframes: list[str] | None = None,
    grid_search: bool = False,
    c_values=None,
    model_type: str = "logreg",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    incremental: bool = False,
    sequence_length: int = 5,
    corr_pairs=None,
    corr_window: int = 5,
    extra_price_series=None,
    optuna_trials: int = 0,
    regress_sl_tp: bool = False,
    early_stop: bool = False,
    encoder_file: Path | None = None,
    cache_features: bool = False,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    calibration: str | None = None,
    stack_models: list[str] | None = None,
    prune_threshold: float = 0.0,
    prune_warn: float = 0.5,
    lite_mode: bool = False,
    compress_model: bool = False,
    regime_model_file: Path | None = None,
    moe: bool = False,
):
    """Train a simple classifier model from the log directory."""
    if lite_mode:
        regime_model = None
        if regime_model_file and regime_model_file.exists():
            with open(regime_model_file) as f:
                regime_model = json.load(f)
        _train_lite_mode(
            data_dir,
            out_dir,
            use_sma=use_sma,
            sma_window=sma_window,
            use_rsi=use_rsi,
            rsi_period=rsi_period,
            use_macd=use_macd,
            use_atr=use_atr,
            atr_period=atr_period,
            use_bollinger=use_bollinger,
            boll_window=boll_window,
            use_stochastic=use_stochastic,
            use_adx=use_adx,
            use_volume=use_volume,
            volatility_series=volatility_series,
            corr_pairs=corr_pairs,
            corr_window=corr_window,
            extra_price_series=extra_price_series,
            calendar_events=calendar_events,
            event_window=event_window,
            encoder_file=encoder_file,
            compress_model=compress_model,
            regime_model=regime_model,
        )
        return
    if optuna_trials > 0:
        try:
            import optuna  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logging.warning(
                "optuna is not installed; skipping hyperparameter search"
            )
            optuna_trials = 0

    cache_file = out_dir / "feature_cache.npz"

    existing_model = None
    last_event_id = 0
    if incremental:
        model_file = out_dir / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found for incremental training")
        with open(model_file) as f:
            existing_model = json.load(f)
            last_event_id = int(existing_model.get("last_event_id", 0))

    features = labels = sl_targets = tp_targets = hours = None
    loaded_from_cache = False
    data_commits: list[str] = []
    data_checksums: list[str] = []
    if cache_features and incremental and cache_file.exists() and existing_model is not None:
        try:
            cached = np.load(cache_file, allow_pickle=True)
            cached_names = list(cached.get("feature_names", []))
            if cached_names == existing_model.get("feature_names", []):
                features = [dict(x) for x in cached["feature_dicts"]]
                labels = cached["labels"]
                sl_targets = cached["sl_targets"]
                tp_targets = cached["tp_targets"]
                hours = cached.get("hours")
                loaded_from_cache = True
        except Exception:
            features = None

    if features is None:
        if lite_mode:
            rows_iter, data_commits, data_checksums = _load_logs(
                data_dir, lite_mode=True
            )
            encoder = None
            if encoder_file is not None and encoder_file.exists():
                with open(encoder_file) as f:
                    encoder = json.load(f)
            features = []
            labels_list: list[np.ndarray] = []
            sl_list: list[np.ndarray] = []
            tp_list: list[np.ndarray] = []
            hours_list: list[np.ndarray] = []
            for chunk in rows_iter:
                (
                    f_chunk,
                    l_chunk,
                    sl_chunk,
                    tp_chunk,
                    h_chunk,
                ) = _extract_features(
                    chunk.to_dict("records"),
                    use_sma=use_sma,
                    sma_window=sma_window,
                    use_rsi=use_rsi,
                    rsi_period=rsi_period,
                    use_macd=use_macd,
                    use_atr=use_atr,
                    atr_period=atr_period,
                    use_bollinger=use_bollinger,
                    boll_window=boll_window,
                    use_stochastic=use_stochastic,
                    use_adx=use_adx,
                    use_volume=use_volume,
                    volatility=volatility_series,
                    higher_timeframes=higher_timeframes,
                    corr_pairs=corr_pairs,
                    corr_window=corr_window,
                    extra_price_series=extra_price_series,
                    encoder=encoder,
                    calendar_events=calendar_events,
                    event_window=event_window,
                )
                features.extend(f_chunk)
                labels_list.append(l_chunk)
                sl_list.append(sl_chunk)
                tp_list.append(tp_chunk)
                hours_list.append(h_chunk)
            labels = np.concatenate(labels_list) if labels_list else np.array([])
            sl_targets = np.concatenate(sl_list) if sl_list else np.array([])
            tp_targets = np.concatenate(tp_list) if tp_list else np.array([])
            hours = (
                np.concatenate(hours_list) if hours_list else np.array([], dtype=int)
            )
        else:
            rows_df, data_commits, data_checksums = _load_logs(data_dir)
            if "event_id" in rows_df.columns:
                max_id = pd.to_numeric(rows_df["event_id"], errors="coerce").max()
                if not pd.isna(max_id):
                    last_event_id = max(last_event_id, int(max_id))
            encoder = None
            if encoder_file is not None and encoder_file.exists():
                with open(encoder_file) as f:
                    encoder = json.load(f)
            (
                features,
                labels,
                sl_targets,
                tp_targets,
                hours,
            ) = _extract_features(
                rows_df.to_dict("records"),
                use_sma=use_sma,
                sma_window=sma_window,
                use_rsi=use_rsi,
                rsi_period=rsi_period,
                use_macd=use_macd,
                use_atr=use_atr,
                atr_period=atr_period,
                use_bollinger=use_bollinger,
                boll_window=boll_window,
                use_stochastic=use_stochastic,
                use_adx=use_adx,
                use_volume=use_volume,
                volatility=volatility_series,
                higher_timeframes=higher_timeframes,
                corr_pairs=corr_pairs,
                corr_window=corr_window,
                extra_price_series=extra_price_series,
                encoder=encoder,
                calendar_events=calendar_events,
                event_window=event_window,
            )
    else:
        encoder = None
        if encoder_file is not None and encoder_file.exists():
            with open(encoder_file) as f:
                encoder = json.load(f)
        if loaded_from_cache:
            rows_df, _, _ = _load_logs(data_dir)
            if "event_id" in rows_df.columns:
                max_id = pd.to_numeric(rows_df["event_id"], errors="coerce").max()
                if not pd.isna(max_id):
                    last_event_id = max(last_event_id, int(max_id))

    if not features:
        raise ValueError(f"No training data found in {data_dir}")
    regime_info = None
    reg_centers = None
    if regime_model_file and regime_model_file.exists():
        with open(regime_model_file) as f:
            regime_info = json.load(f)
        vec_reg = DictVectorizer(sparse=False)
        vec_reg.fit([{n: 0.0} for n in regime_info.get("feature_names", [])])
        Xr = vec_reg.transform(features)
        reg_mean = np.array(regime_info.get("mean", []), dtype=float)
        reg_std = np.array(regime_info.get("std", []), dtype=float)
        reg_std[reg_std == 0] = 1.0
        Xr = (Xr - reg_mean) / reg_std
        reg_centers = np.array(regime_info.get("centers", []), dtype=float)
        if reg_centers.size:
            dists = ((Xr[:, None, :] - reg_centers[None, :, :]) ** 2).sum(axis=2)
            regimes = dists.argmin(axis=1)
            for i, r in enumerate(regimes):
                features[i][f"regime_{int(r)}"] = 1.0
    ae_info = None
    ae_feature_order = ["price", "sl", "tp", "lots", "spread", "slippage"]
    ae_matrix = np.array(
        [[float(f.get(k, 0.0)) for k in ae_feature_order] for f in features],
        dtype=float,
    )
    ae_hidden = 4
    if ae_matrix.shape[0] >= 10:
        ae_mean = ae_matrix.mean(axis=0)
        ae_std = ae_matrix.std(axis=0)
        ae_std[ae_std == 0] = 1.0
        ae_scaled = (ae_matrix - ae_mean) / ae_std
        ae_model = MLPRegressor(hidden_layer_sizes=(ae_hidden,), max_iter=200, random_state=42)
        ae_model.fit(ae_scaled, ae_scaled)
        recon = ae_model.predict(ae_scaled)
        ae_errors = np.mean((ae_scaled - recon) ** 2, axis=1)
        ae_threshold = float(np.percentile(ae_errors, 95))
        mask = ae_errors <= ae_threshold
        if mask.sum() > 0:
            features = [features[i] for i, m in enumerate(mask) if m]
            labels = labels[mask]
            sl_targets = sl_targets[mask]
            tp_targets = tp_targets[mask]
            hours = hours[mask]
        ae_info = {
            "weights": [w.astype(np.float32).tolist() for w in ae_model.coefs_],
            "bias": [b.astype(np.float32).tolist() for b in ae_model.intercepts_],
            "mean": ae_mean.astype(np.float32).tolist(),
            "std": ae_std.astype(np.float32).tolist(),
            "threshold": ae_threshold,
            "feature_order": ae_feature_order,
        }
    hidden_size = 8
    logreg_C = 1.0
    best_trial = None
    study = None
    sl_coef = []
    tp_coef = []
    sl_inter = 0.0
    tp_inter = 0.0

    if existing_model is not None:
        vec = DictVectorizer(sparse=False)
        vec.fit([{name: 0.0} for name in existing_model.get("feature_names", [])])
    else:
        vec = DictVectorizer(sparse=False)

    if len(labels) < 5 or len(np.unique(labels)) < 2:
        # Not enough data to create a meaningful split
        feat_train, y_train = features, labels
        feat_val, y_val = [], np.array([])
        sl_train = sl_targets
        sl_val = np.array([])
        tp_train = tp_targets
        tp_val = np.array([])
        hours_train = hours
        hours_val = np.array([])
    else:
        tscv = TimeSeriesSplit(n_splits=min(5, len(labels) - 1))
        # select the final chronological split for validation
        train_idx, val_idx = list(tscv.split(features))[-1]
        feat_train = [features[i] for i in train_idx]
        feat_val = [features[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        sl_train = sl_targets[train_idx]
        sl_val = sl_targets[val_idx]
        tp_train = tp_targets[train_idx]
        tp_val = tp_targets[val_idx]
        hours_train = hours[train_idx]
        hours_val = hours[val_idx]

        # if the training split ended up with only one class, fall back to using
        # all data for training so the model can be fit
        if len(np.unique(y_train)) < 2:
            feat_train, y_train = features, labels
            feat_val, y_val = [], np.array([])
            hours_train = hours
            hours_val = np.array([])

    sample_weight = None
    if model_type == "logreg" and not grid_search:
        sample_weight = np.array(
            [abs(f.get("profit", f.get("lots", 1.0))) for f in feat_train],
            dtype=float,
        )

    feat_train_clf = [dict(f) for f in feat_train]
    feat_val_clf = [dict(f) for f in feat_val]
    for f in feat_train_clf:
        f.pop("profit", None)
    for f in feat_val_clf:
        f.pop("profit", None)

    feat_train_reg = [dict(f) for f in feat_train_clf]
    feat_val_reg = [dict(f) for f in feat_val_clf]
    for f in feat_train_reg:
        f["sl_dist"] = 0.0
        f["tp_dist"] = 0.0
    for f in feat_val_reg:
        f["sl_dist"] = 0.0
        f["tp_dist"] = 0.0

    if existing_model is not None:
        X_train = vec.transform(feat_train_clf)
    else:
        X_train = vec.fit_transform(feat_train_clf)
    if feat_val_clf:
        X_val = vec.transform(feat_val_clf)
    else:
        X_val = np.empty((0, X_train.shape[1]))

    feature_names = vec.get_feature_names_out().tolist()
    X_train_reg = vec.transform(feat_train_reg)
    X_val_reg = (
        vec.transform(feat_val_reg) if feat_val_reg else np.empty((0, X_train_reg.shape[1]))
    )

    if cache_features and not loaded_from_cache:
        feature_names_cache = feature_names
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            feature_dicts=np.array(features, dtype=object),
            labels=labels,
            sl_targets=sl_targets,
            tp_targets=tp_targets,
            hours=hours,
            feature_names=np.array(feature_names_cache),
        )

    optuna_threshold = None
    if optuna_trials > 0:
        available_models = ["logreg", "random_forest", "xgboost", "nn"]
        try:
            import importlib.util

            if importlib.util.find_spec("xgboost") is None:
                available_models.remove("xgboost")
        except Exception:
            if "xgboost" in available_models:
                available_models.remove("xgboost")

        def _objective(trial):
            model_choice = trial.suggest_categorical("model_type", available_models)
            max_feats = min(len(feature_names), 10)
            sel_idx = []
            for i, name in enumerate(feature_names[:max_feats]):
                if trial.suggest_categorical(f"f_{name}", [True, False]):
                    sel_idx.append(i)
            if not sel_idx:
                sel_idx = list(range(max_feats))
            sel_idx += list(range(max_feats, len(feature_names)))
            X_tr = X_train[:, sel_idx]
            X_v = X_val[:, sel_idx]
            if model_choice == "logreg":
                c = trial.suggest_float("C", 1e-3, 10.0, log=True)
                clf = LogisticRegression(max_iter=200, C=c)
                clf.fit(X_tr, y_train)
            elif model_choice == "random_forest":
                est = trial.suggest_int("n_estimators", 50, 200)
                depth = trial.suggest_int("max_depth", 2, 8)
                clf = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=42)
                clf.fit(X_tr, y_train)
            elif model_choice == "xgboost":
                from xgboost import XGBClassifier  # type: ignore

                est = trial.suggest_int("n_estimators", 50, 300)
                lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                clf = XGBClassifier(
                    n_estimators=est,
                    learning_rate=lr,
                    max_depth=depth,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
                clf.fit(X_tr, y_train)
            else:
                h = trial.suggest_int("hidden_size", 4, 64)
                try:
                    from tensorflow import keras  # type: ignore

                    clf = keras.Sequential(
                        [
                            keras.layers.Input(shape=(X_tr.shape[1],)),
                            keras.layers.Dense(h, activation="relu"),
                            keras.layers.Dense(1, activation="sigmoid"),
                        ]
                    )
                    clf.compile(optimizer="adam", loss="binary_crossentropy")
                    clf.fit(X_tr, y_train, epochs=50, verbose=0)
                except Exception:
                    clf = MLPClassifier(
                        hidden_layer_sizes=(h,), max_iter=500, random_state=42
                    )
                    clf.fit(X_tr, y_train)

            if hasattr(clf, "predict_proba"):
                val_proba = clf.predict_proba(X_v)[:, 1] if len(y_val) > 0 else np.empty(0)
            else:
                val_proba = clf.predict(X_v).reshape(-1) if len(y_val) > 0 else np.empty(0)

            thr = trial.suggest_float("threshold", 0.3, 0.7)
            trial.set_user_attr("features", [feature_names[i] for i in sel_idx])
            if len(y_val) > 0:
                preds = (val_proba >= thr).astype(int)
                return accuracy_score(y_val, preds)
            return 0.0

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(_objective, n_trials=optuna_trials)
        best_trial = study.best_trial
        model_type = best_trial.params.get("model_type", model_type)
        optuna_threshold = float(best_trial.params.get("threshold", 0.5))
        max_feats = min(len(feature_names), 10)
        sel_idx = [
            i
            for i, name in enumerate(feature_names[:max_feats])
            if best_trial.params.get(f"f_{name}", True)
        ]
        if not sel_idx:
            sel_idx = list(range(max_feats))
        sel_idx += list(range(max_feats, len(feature_names)))
        selected_indices = sel_idx
        X_train = X_train[:, sel_idx]
        X_val = X_val[:, sel_idx]
        X_train_reg = X_train_reg[:, sel_idx]
        X_val_reg = X_val_reg[:, sel_idx]
        feature_names = [feature_names[i] for i in sel_idx]
        if model_type == "logreg" and "C" in best_trial.params:
            logreg_C = float(best_trial.params["C"])
        elif model_type == "random_forest":
            n_estimators = int(best_trial.params["n_estimators"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "xgboost":
            n_estimators = int(best_trial.params["n_estimators"])
            learning_rate = float(best_trial.params["learning_rate"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "nn":
            hidden_size = int(best_trial.params["hidden_size"])

    # statistics for feature scaling
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    if moe:
        from sklearn.preprocessing import LabelEncoder

        all_feat_clf = [dict(f) for f in features]
        for f in all_feat_clf:
            f.pop("profit", None)
        X_all = vec.transform(all_feat_clf)
        symbols_all = np.array([f.get("symbol", "") for f in features])
        le = LabelEncoder()
        sym_idx = le.fit_transform(symbols_all)
        gating_clf = LogisticRegression(max_iter=200, multi_class="multinomial")
        gating_clf.fit(X_all, sym_idx)
        gating_coef = gating_clf.coef_.ravel().astype(np.float32).tolist()
        gating_inter = gating_clf.intercept_.astype(np.float32).tolist()

        expert_models = []
        for idx, sym in enumerate(le.classes_):
            mask = symbols_all == sym
            if not mask.any():
                continue
            X_sym = X_all[mask]
            y_sym = labels[mask]
            sw = np.array(
                [abs(features[i].get("profit", features[i].get("lots", 1.0))) for i in np.where(mask)[0]],
                dtype=float,
            )
            exp_clf = LogisticRegression(max_iter=200)
            exp_clf.fit(X_sym, y_sym, sample_weight=sw)
            expert_models.append(
                {
                    "coefficients": exp_clf.coef_[0].astype(np.float32).tolist(),
                    "intercept": float(exp_clf.intercept_[0]),
                    "classes": [int(c) for c in exp_clf.classes_],
                    "symbol": sym,
                }
            )

        model = {
            "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": feature_names,
            "model_type": "moe_logreg",
            "gating_coefficients": gating_coef,
            "gating_intercepts": gating_inter,
            "gating_classes": le.classes_.tolist(),
            "session_models": expert_models,
            "last_event_id": int(last_event_id),
            "mean": feature_mean.astype(np.float32).tolist(),
            "std": feature_std.astype(np.float32).tolist(),
        }
        if ae_info:
            model["autoencoder"] = ae_info
        if data_commits:
            model["data_commit"] = ",".join(sorted(set(data_commits)))
        if data_checksums:
            model["data_checksum"] = ",".join(sorted(set(data_checksums)))
        if calendar_events:
            model["calendar_events"] = [
                [dt.isoformat(), float(imp)] for dt, imp in calendar_events
            ]
            model["event_window"] = float(event_window)
        model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
        open_func = gzip.open if compress_model else open
        with open_func(model_path, "wt") as f:
            json.dump(model, f)
        print(f"Model written to {model_path}")
        return

    if stack_models:
        estimators = []
        for mt in stack_models:
            if mt == "logreg":
                estimators.append(("logreg", LogisticRegression(max_iter=200)))
            elif mt == "random_forest":
                estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))
            elif mt == "xgboost":
                try:
                    from xgboost import XGBClassifier  # type: ignore

                    estimators.append(
                        (
                            "xgb",
                            XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                eval_metric="logloss",
                                use_label_encoder=False,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "xgboost is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("xgb", LogisticRegression(max_iter=200)))
            elif mt == "lgbm":
                try:
                    from lightgbm import LGBMClassifier  # type: ignore

                    estimators.append(
                        (
                            "lgbm",
                            LGBMClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "lightgbm is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("lgbm", LogisticRegression(max_iter=200)))
            elif mt == "catboost":
                try:
                    from catboost import CatBoostClassifier  # type: ignore

                    estimators.append(
                        (
                            "cat",
                            CatBoostClassifier(
                                iterations=n_estimators,
                                learning_rate=learning_rate,
                                depth=max_depth,
                                verbose=False,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "catboost is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("cat", LogisticRegression(max_iter=200)))
            elif mt == "nn":
                estimators.append(
                    ("nn", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42))
                )
        final_est = LogisticRegression(max_iter=200)
        clf = StackingClassifier(estimators=estimators, final_estimator=final_est, stack_method="predict_proba")
        clf.fit(X_train, y_train)
        train_proba_raw = clf.predict_proba(X_train)[:, 1]
        val_proba_raw = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
        model_type = "stack"
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        train_proba_raw = clf.predict_proba(X_train)[:, 1]
        val_proba_raw = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore

            clf = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "xgboost is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            clf = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "lightgbm is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier  # type: ignore

            clf = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                verbose=False,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "catboost is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "nn":
        try:
            from tensorflow import keras  # type: ignore

            model_nn = keras.Sequential(
                [
                    keras.layers.Input(shape=(X_train.shape[1],)),
                    keras.layers.Dense(hidden_size, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
        except Exception:
            logging.warning(
                "TensorFlow not available; using MLPClassifier instead"
            )
            clf = MLPClassifier(
                hidden_layer_sizes=(hidden_size,), max_iter=500, random_state=42
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
    elif model_type == "lstm":
        try:
            from tensorflow import keras  # type: ignore
        except Exception:
            logging.warning(
                "TensorFlow is required for LSTM model; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
        else:
            seq_len = sequence_length
            X_all = (
                vec.fit_transform(features)
                if existing_model is None
                else vec.transform(features)
            )
            sequences = []
            for i in range(len(X_all)):
                start = max(0, i - seq_len + 1)
                seq = X_all[start : i + 1]
                if seq.shape[0] < seq_len:
                    pad = np.zeros((seq_len - seq.shape[0], X_all.shape[1]))
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
            X_all_seq = np.array(sequences)
            if len(labels) < 5 or len(np.unique(labels)) < 2:
                X_train_seq, y_train = X_all_seq, labels
                X_val_seq, y_val = (
                    np.empty((0, seq_len, X_all.shape[1])),
                    np.array([]),
                )
            else:
                X_train_seq, X_val_seq, y_train, y_val = train_test_split(
                    X_all_seq,
                    labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels,
                )
            model_nn = keras.Sequential(
                [
                    keras.layers.Input(shape=(seq_len, X_all.shape[1])),
                    keras.layers.LSTM(8),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train_seq,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train_seq).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val_seq).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
    elif model_type == "transformer":
        try:
            from tensorflow import keras  # type: ignore
        except Exception:
            logging.warning(
                "TensorFlow is required for transformer model; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
        else:
            seq_len = sequence_length
            X_all = (
                vec.fit_transform(features)
                if existing_model is None
                else vec.transform(features)
            )
            sequences = []
            for i in range(len(X_all)):
                start = max(0, i - seq_len + 1)
                seq = X_all[start : i + 1]
                if seq.shape[0] < seq_len:
                    pad = np.zeros((seq_len - seq.shape[0], X_all.shape[1]))
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
            X_all_seq = np.array(sequences)
            if len(labels) < 5 or len(np.unique(labels)) < 2:
                X_train_seq, y_train = X_all_seq, labels
                X_val_seq, y_val = (
                    np.empty((0, seq_len, X_all.shape[1])),
                    np.array([]),
                )
            else:
                X_train_seq, X_val_seq, y_train, y_val = train_test_split(
                    X_all_seq,
                    labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels,
                )
            inp = keras.layers.Input(shape=(seq_len, X_all.shape[1]))
            att = keras.layers.MultiHeadAttention(num_heads=1, key_dim=X_all.shape[1])(inp, inp)
            pooled = keras.layers.GlobalAveragePooling1D()(att)
            out = keras.layers.Dense(1, activation="sigmoid")(pooled)
            model_nn = keras.Model(inp, out)
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train_seq,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train_seq).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val_seq).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
    else:
        if grid_search:
            if c_values is None:
                c_values = [0.01, 0.1, 1.0, 10.0]
            param_grid = {"C": c_values}
            gs = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
            gs.fit(X_train, y_train)
            clf = gs.best_estimator_
        else:
            if incremental:
                clf = SGDClassifier(loss="log_loss", alpha=1.0 / logreg_C)
                if existing_model is not None:
                    clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
                    clf.coef_ = np.array([existing_model.get("coefficients", [])])
                    clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
                classes = getattr(clf, "classes_", np.array([0, 1]))
                batch_size = 1000
                for start in range(0, X_train.shape[0], batch_size):
                    end = start + batch_size
                    sw = sample_weight[start:end] if sample_weight is not None else None
                    clf.partial_fit(
                        X_train[start:end], y_train[start:end], classes=classes, sample_weight=sw
                    )
                train_proba_raw = clf.predict_proba(X_train)[:, 1]
                val_proba_raw = (
                    clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
                )
            else:
                clf = LogisticRegression(max_iter=200, C=logreg_C, warm_start=existing_model is not None)
                if existing_model is not None:
                    clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
                    clf.coef_ = np.array([existing_model.get("coefficients", [])])
                    clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
                clf.fit(X_train, y_train, sample_weight=sample_weight)
                train_proba_raw = clf.predict_proba(X_train)[:, 1]
                val_proba_raw = (
                    clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
                )

    train_proba = train_proba_raw
    val_proba = val_proba_raw
    cal_coef = 1.0
    cal_inter = 0.0
    if calibration is not None and len(y_val) > 0:
        calibrator = CalibratedClassifierCV(clf, cv="prefit", method=calibration)
        calibrator.fit(X_val, y_val)
        train_proba = calibrator.predict_proba(X_train)[:, 1]
        val_proba = calibrator.predict_proba(X_val)[:, 1]
        if calibration == "sigmoid":
            cal_lr = calibrator.calibrated_classifiers_[0].calibrator
            cal_coef = float(cal_lr.coef_[0][0])
            cal_inter = float(cal_lr.intercept_[0])
    

    if regress_sl_tp:
        from sklearn.linear_model import LinearRegression

        reg_sl = LinearRegression()
        reg_sl.fit(X_train_reg, sl_train)
        reg_tp = LinearRegression()
        reg_tp.fit(X_train_reg, tp_train)
        sl_coef = reg_sl.coef_
        sl_inter = reg_sl.intercept_
        tp_coef = reg_tp.coef_
        tp_inter = reg_tp.intercept_

    if len(y_val) > 0:
        if optuna_threshold is not None:
            threshold = optuna_threshold
        else:
            threshold, _ = _best_threshold(y_val, val_proba)
        val_preds = (val_proba >= threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_preds))
    else:
        threshold = optuna_threshold if optuna_threshold is not None else 0.5
        val_acc = float("nan")
    train_preds = (train_proba >= threshold).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))

    hourly_thresholds = None
    if len(y_val) > 0:
        hours_val_arr = np.array(hours_val, dtype=int)
        hourly_thresholds = []
        for h in range(24):
            idx = np.where(hours_val_arr == h)[0]
            if len(idx) > 0:
                t, _ = _best_threshold(y_val[idx], val_proba[idx])
            else:
                t = threshold
            hourly_thresholds.append(float(t))

    # Compute SHAP feature importance on the training set
    keep_idx = list(range(len(feature_names)))
    try:
        import shap  # type: ignore

        if model_type == "logreg":
            explainer = shap.LinearExplainer(clf, X_train)
            shap_values = explainer.shap_values(X_train)
        else:
            explainer = shap.Explainer(clf, X_train)
            shap_values = explainer(X_train).values
        importances = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, importances.tolist()))
    except Exception:  # pragma: no cover - shap optional
        importances = np.array([])
        feature_importance = {}

    if prune_threshold > 0.0 and feature_importance:
        keep_idx = [i for i, name in enumerate(feature_names) if feature_importance.get(name, 0.0) >= prune_threshold]
        removed_ratio = 1 - len(keep_idx) / len(feature_names)
        if removed_ratio > prune_warn:
            logging.warning("Pruning removed %.1f%% of features", removed_ratio * 100)
        if len(keep_idx) < len(feature_names):
            X_train = X_train[:, keep_idx]
            if X_val.shape[0] > 0:
                X_val = X_val[:, keep_idx]
            X_train_reg = X_train_reg[:, keep_idx]
            if X_val_reg.shape[0] > 0:
                X_val_reg = X_val_reg[:, keep_idx]
            feature_mean = feature_mean[keep_idx]
            feature_std = feature_std[keep_idx]
            feature_names = [feature_names[i] for i in keep_idx]
            selected_indices = [selected_indices[i] for i in keep_idx]

            clf.fit(X_train, y_train, sample_weight=sample_weight)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            train_proba = train_proba_raw
            val_proba = val_proba_raw
            if calibration is not None and len(y_val) > 0:
                calibrator = CalibratedClassifierCV(clf, cv="prefit", method=calibration)
                calibrator.fit(X_val, y_val)
                train_proba = calibrator.predict_proba(X_train)[:, 1]
                val_proba = calibrator.predict_proba(X_val)[:, 1]
                if calibration == "sigmoid":
                    cal_lr = calibrator.calibrated_classifiers_[0].calibrator
                    cal_coef = float(cal_lr.coef_[0][0])
                    cal_inter = float(cal_lr.intercept_[0])
            if regress_sl_tp:
                reg_sl.fit(X_train_reg, sl_train)
                reg_tp.fit(X_train_reg, tp_train)
                sl_coef = reg_sl.coef_
                sl_inter = reg_sl.intercept_
                tp_coef = reg_tp.coef_
                tp_inter = reg_tp.intercept_
            if len(y_val) > 0:
                if optuna_threshold is not None:
                    threshold = optuna_threshold
                else:
                    threshold, _ = _best_threshold(y_val, val_proba)
                val_preds = (val_proba >= threshold).astype(int)
                val_acc = float(accuracy_score(y_val, val_preds))
            else:
                threshold = optuna_threshold if optuna_threshold is not None else 0.5
                val_acc = float("nan")
            train_preds = (train_proba >= threshold).astype(int)
            train_acc = float(accuracy_score(y_train, train_preds))

            try:
                if model_type == "logreg":
                    explainer = shap.LinearExplainer(clf, X_train)
                    shap_values = explainer.shap_values(X_train)
                else:
                    explainer = shap.Explainer(clf, X_train)
                    shap_values = explainer(X_train).values
                importances = np.abs(shap_values).mean(axis=0)
                feature_importance = dict(zip(feature_names, importances.tolist()))
            except Exception:  # pragma: no cover
                feature_importance = {}


    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": model_type,
        "weighted": sample_weight is not None,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "threshold": threshold,
        # main accuracy metric is validation performance when available
        "accuracy": val_acc,
        "num_samples": int(labels.shape[0]) + (int(existing_model.get("num_samples", 0)) if existing_model else 0),
        "last_event_id": int(last_event_id),
        "feature_importance": feature_importance,
        "mean": feature_mean.astype(np.float32).tolist(),
        "std": feature_std.astype(np.float32).tolist(),
    }
    if ae_info:
        model["autoencoder"] = ae_info
    if regime_info is not None and reg_centers is not None:
        model["regime_centers"] = reg_centers.astype(float).tolist()
        model["regime_feature_names"] = regime_info.get("feature_names", [])
    if data_commits:
        model["data_commit"] = ",".join(sorted(set(data_commits)))
    if data_checksums:
        model["data_checksum"] = ",".join(sorted(set(data_checksums)))
    if calendar_events:
        model["calendar_events"] = [
            [dt.isoformat(), float(imp)] for dt, imp in calendar_events
        ]
        model["event_window"] = float(event_window)
    if encoder is not None:
        model["encoder_weights"] = encoder.get("weights")
        model["encoder_window"] = encoder.get("window")
        if "centers" in encoder:
            model["encoder_centers"] = encoder.get("centers")
    if best_trial is not None and study is not None:
        model["optuna_best_params"] = best_trial.params
        model["optuna_best_score"] = float(best_trial.value)
        model["optuna_study"] = {"n_trials": len(study.trials)}
        model["optuna_trials"] = [
            {"params": t.params, "value": float(t.value)} for t in study.trials
        ]
    if hourly_thresholds is not None:
        model["hourly_thresholds"] = hourly_thresholds
    if calibration is not None:
        model["calibration_method"] = calibration
        if calibration == "sigmoid":
            model["calibration_coef"] = cal_coef
            model["calibration_intercept"] = cal_inter
    if stack_models:
        model["stack_models"] = stack_models

    if model_type == "logreg":
        model["coefficients"] = clf.coef_[0].astype(np.float32).tolist()
        model["intercept"] = float(clf.intercept_[0])
        model["classes"] = [int(c) for c in clf.classes_]
    elif model_type == "xgboost":
        # approximate tree ensemble with linear model for MQL4 export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        # lookup probabilities per trading hour for simple export
        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "lgbm":
        # approximate boosting model with linear regression for export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "catboost":
        # approximate boosting model with linear regression for export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "stack":
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])
    elif model_type == "nn":
        if hasattr(clf, "get_weights"):
            weights = [w.tolist() for w in clf.get_weights()]
        else:
            weights = [
                clf.coefs_[0].tolist(),
                clf.intercepts_[0].tolist(),
                clf.coefs_[1].tolist(),
                clf.intercepts_[1].tolist(),
            ]
        model["nn_weights"] = weights
        model["hidden_size"] = len(weights[1]) if weights else 0
    elif model_type == "lstm":
        weights = [w.tolist() for w in clf.get_weights()]
        model["lstm_weights"] = weights
        model["sequence_length"] = sequence_length
        model["hidden_size"] = len(weights[1]) // 4 if weights else 0
    elif model_type == "transformer":
        weights = [w.tolist() for w in clf.get_weights()]
        model["transformer_weights"] = weights
        model["sequence_length"] = sequence_length

    if regress_sl_tp:
        model["sl_coefficients"] = sl_coef.astype(np.float32).tolist()
        model["sl_intercept"] = float(sl_inter)
        model["tp_coefficients"] = tp_coef.astype(np.float32).tolist()
        model["tp_intercept"] = float(tp_inter)

    # Optional RL refinement
    if (
        HAS_SB3
        and _has_sufficient_gpu()
        and _has_sufficient_ram()
        and "coefficients" in model
        and "intercept" in model
    ):
        try:
            temp_model = out_dir / "model_supervised.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(temp_model, "w") as f_tmp:
                json.dump(model, f_tmp)
            rl_out = out_dir / "rl_tmp"
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("train_rl_agent.py")),
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(rl_out),
                "--algo",
                "qlearn",
                "--training-steps",
                "20",
                "--start-model",
                str(temp_model),
            ]
            if compress_model:
                cmd.append("--compress-model")
            subprocess.run(cmd, check=True)
            rl_model_path = rl_out / (
                "model.json.gz" if compress_model else "model.json"
            )
            open_rl = gzip.open if compress_model else open
            with open_rl(rl_model_path, "rt") as f_rl:
                rl_model = json.load(f_rl)
            if "coefficients" in rl_model and "intercept" in rl_model:
                model["coefficients"] = rl_model["coefficients"]
                model["intercept"] = rl_model["intercept"]
            if "q_weights" in rl_model:
                model["q_weights"] = rl_model["q_weights"]
            if "q_intercepts" in rl_model:
                model["q_intercepts"] = rl_model["q_intercepts"]
            model["rl_steps"] = rl_model.get("training_steps")
            model["rl_reward"] = rl_model.get("avg_reward")
        except Exception as exc:  # pragma: no cover - optional RL errors
            logging.warning("RL refinement failed: %s", exc)
        finally:
            try:
                temp_model.unlink()
            except Exception:
                pass

    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {model_path}")
    _export_onnx(clf, model.get("feature_names", []), out_dir)

    if "coefficients" in model and "intercept" in model:
        w = np.array(model["coefficients"], dtype=float)
        b = float(model["intercept"])
        init = {
            "weights": [
                (w / 2.0).tolist(),
                (-w / 2.0).tolist(),
            ],
            "intercepts": [b / 2.0, -b / 2.0],
            "feature_names": model.get("feature_names", []),
        }
        with open(out_dir / "policy_init.json", "w") as f:
            json.dump(init, f, indent=2)
        print(f"Initial policy written to {out_dir / 'policy_init.json'}")

    print(f"Validation accuracy: {val_acc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--use-sma', action='store_true', help='include moving average feature')
    p.add_argument('--sma-window', type=int, default=5)
    p.add_argument('--use-rsi', action='store_true', help='include RSI feature')
    p.add_argument('--rsi-period', type=int, default=14)
    p.add_argument('--use-macd', action='store_true', help='include MACD feature')
    p.add_argument('--use-atr', action='store_true', help='include ATR feature')
    p.add_argument('--use-bollinger', action='store_true', help='include Bollinger Bands feature')
    p.add_argument('--use-stochastic', action='store_true', help='include Stochastic Oscillator feature')
    p.add_argument('--use-adx', action='store_true', help='include ADX feature')
    p.add_argument(
        '--higher-timeframes',
        help='comma separated higher timeframes e.g. H1,H4',
    )
    p.add_argument('--calendar-file', help='CSV file with columns time,impact for events')
    p.add_argument('--event-window', type=float, default=60.0, help='minutes around events to flag')
    p.add_argument('--volatility-file', help='JSON file with precomputed volatility')
    p.add_argument('--grid-search', action='store_true', help='enable grid search with cross-validation')
    p.add_argument('--c-values', type=float, nargs='*')
    p.add_argument('--sequence-length', type=int, default=5, help='sequence length for LSTM/transformer models')
    p.add_argument('--n-estimators', type=int, default=100, help='number of boosting rounds')
    p.add_argument('--learning-rate', type=float, default=0.1, help='learning rate for boosted trees')
    p.add_argument('--max-depth', type=int, default=3, help='tree depth for boosting models')
    p.add_argument('--incremental', action='store_true', help='update existing model.json')
    p.add_argument('--start-event-id', type=int, default=0, help='only load rows with event_id greater than this value from SQLite logs')
    p.add_argument('--resume', action='store_true', help='resume from last processed event_id in existing model.json')
    p.add_argument('--cache-features', action='store_true', help='reuse cached feature matrix')
    p.add_argument('--corr-symbols', help='comma separated correlated symbol pairs e.g. EURUSD:USDCHF')
    p.add_argument('--corr-window', type=int, default=5, help='window for correlation calculations')
    p.add_argument(
        '--optuna-trials',
        type=int,
        default=0,
        help='number of Optuna trials to tune model type, threshold and features',
    )
    p.add_argument('--encoder-file', help='JSON file with pretrained encoder weights')
    p.add_argument('--regress-sl-tp', action='store_true', help='learn SL/TP distance regressors')
    p.add_argument('--early-stop', action='store_true', help='enable early stopping for neural nets')
    p.add_argument('--calibration', choices=['sigmoid', 'isotonic'], help='probability calibration method')
    p.add_argument('--stack', help='comma separated list of model types to stack')
    p.add_argument('--prune-threshold', type=float, default=0.0, help='drop features with SHAP importance below this value')
    p.add_argument('--prune-warn', type=float, default=0.5, help='warn if more than this fraction of features are pruned')
    p.add_argument('--compress-model', action='store_true', help='write model.json.gz')
    p.add_argument('--regime-model', help='JSON file with precomputed regime centers')
    p.add_argument('--moe', action='store_true', help='train mixture-of-experts model per symbol')
    args = p.parse_args()
    global START_EVENT_ID
    if args.resume:
        START_EVENT_ID = _read_last_event_id(Path(args.out_dir))
    else:
        START_EVENT_ID = args.start_event_id
    if args.volatility_file:
        import json
        with open(args.volatility_file) as f:
            vol_data = json.load(f)
    else:
        vol_data = None
    if args.calendar_file:
        events = _load_calendar(Path(args.calendar_file))
    else:
        events = None
    if args.corr_symbols:
        corr_pairs = [tuple(p.split(':')) for p in args.corr_symbols.split(',')]
    else:
        corr_pairs = None
    if args.higher_timeframes:
        higher_tfs = [tf.strip() for tf in args.higher_timeframes.split(',') if tf.strip()]
    else:
        higher_tfs = None
    resources = detect_resources()
    lite_mode = resources["lite_mode"]
    model_type = resources["model_type"]
    optuna_trials = (
        0 if lite_mode else (args.optuna_trials or resources["optuna_trials"])
    )
    train(
        Path(args.data_dir),
        Path(args.out_dir),
        use_sma=args.use_sma,
        sma_window=args.sma_window,
        use_rsi=args.use_rsi,
        rsi_period=args.rsi_period,
        use_macd=args.use_macd,
        use_atr=args.use_atr,
        use_bollinger=args.use_bollinger,
        use_stochastic=args.use_stochastic,
        use_adx=args.use_adx,
        higher_timeframes=higher_tfs,
        volatility_series=vol_data,
        grid_search=args.grid_search,
        c_values=args.c_values,
        model_type=model_type,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        incremental=args.incremental,
        sequence_length=args.sequence_length,
        corr_pairs=corr_pairs,
        corr_window=args.corr_window,
        optuna_trials=optuna_trials,
        regress_sl_tp=args.regress_sl_tp,
        early_stop=args.early_stop,
        encoder_file=Path(args.encoder_file) if args.encoder_file else None,
        cache_features=args.cache_features,
        calendar_events=events,
        event_window=args.event_window,
        calibration=args.calibration,
        stack_models=[s.strip() for s in args.stack.split(',')] if args.stack else None,
        prune_threshold=args.prune_threshold,
        prune_warn=args.prune_warn,
        lite_mode=lite_mode,
        compress_model=args.compress_model,
        regime_model_file=Path(args.regime_model) if args.regime_model else None,
        moe=args.moe,
    )


if __name__ == '__main__':
    main()
