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
from datetime import datetime
import math
from pathlib import Path
from typing import List, Optional
import sqlite3

import pandas as pd

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:  # pragma: no cover - optional dependency
    HAS_OPTUNA = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    HAS_TF = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TF = False

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit


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
        df_logs = pd.read_sql_query("SELECT * FROM logs", conn, parse_dates=["event_time"])
    finally:
        conn.close()

    df_logs.columns = [c.lower() for c in df_logs.columns]

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    if "action" in df_logs.columns:
        df_logs["action"] = df_logs["action"].fillna("").str.upper()
        df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    return df_logs


def _load_logs(data_dir: Path) -> pd.DataFrame:
    """Load log rows from ``data_dir``.

    ``MODIFY`` entries are retained alongside ``OPEN`` and ``CLOSE``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files or a SQLite ``.db`` file.

    Returns
    -------
    pandas.DataFrame
        Parsed rows as a DataFrame.
    """

    if data_dir.suffix == ".db":
        return _load_logs_db(data_dir)

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
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
    ]

    dfs: List[pd.DataFrame] = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        df = pd.read_csv(
            log_file,
            sep=";",
            names=fields,
            header=0,
            parse_dates=["event_time"],
        )
        dfs.append(df)

    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=fields)

    df_logs.columns = [c.lower() for c in df_logs.columns]

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    df_logs["action"] = df_logs["action"].fillna("").str.upper()
    df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    metrics_file = data_dir / "metrics.csv"
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file, sep=";")
        df_metrics.columns = [c.lower() for c in df_metrics.columns]
        if "magic" in df_metrics.columns:
            key_col = "magic"
        elif "model_id" in df_metrics.columns:
            key_col = "model_id"
        else:
            key_col = None

        if key_col is not None:
            df_metrics[key_col] = pd.to_numeric(df_metrics[key_col], errors="coerce").fillna(0).astype(int)
            df_logs["magic"] = pd.to_numeric(df_logs["magic"], errors="coerce").fillna(0).astype(int)
            if key_col == "magic":
                df_logs = df_logs.merge(df_metrics, how="left", on="magic")
            else:
                df_logs = df_logs.merge(df_metrics, how="left", left_on="magic", right_on="model_id")
                df_logs = df_logs.drop(columns=["model_id"])

    return df_logs


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
    use_slippage=False,
    use_volume=False,
    volatility=None,
    use_higher_timeframe=False,
    higher_timeframe="H1",
    *,
    corr_pairs=None,
    extra_price_series=None,
    corr_window: int = 5,
    encoder: dict | None = None,
):
    feature_dicts = []
    labels = []
    sl_targets = []
    tp_targets = []
    prices = []
    macd_state = {}
    tf_prices = []
    tf_macd_state = {}
    tf_macd = 0.0
    tf_macd_sig = 0.0
    tf_last_bin = None
    tf_prev_price = None
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
    tf_minutes = tf_map.get(str(higher_timeframe).upper(), 60)
    tf_secs = tf_minutes * 60
    stoch_state = {}
    adx_state = {}
    price_map = {sym: list(vals) for sym, vals in (extra_price_series or {}).items()}
    enc_window = int(encoder.get("window")) if encoder else 0
    enc_weights = (
        np.array(encoder.get("weights", []), dtype=float) if encoder else np.empty((0, 0))
    )
    enc_centers = (
        np.array(encoder.get("centers", []), dtype=float) if encoder else np.empty((0, 0))
    )
    for r in rows:
        if r.get("action", "").upper() != "OPEN":
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

        price = float(r.get("price", 0) or 0)
        sl = float(r.get("sl", 0) or 0)
        tp = float(r.get("tp", 0) or 0)
        lots = float(r.get("lots", 0) or 0)
        profit = float(r.get("profit", 0) or 0)

        tf_bin = int(t.timestamp() // tf_secs)
        if tf_last_bin is None:
            tf_last_bin = tf_bin
        elif tf_bin != tf_last_bin:
            if tf_prev_price is not None:
                tf_prices.append(tf_prev_price)
                if use_higher_timeframe and use_macd:
                    tf_macd, tf_macd_sig = _macd_update(tf_macd_state, tf_prev_price)
            tf_last_bin = tf_bin
        tf_prev_price = price

        symbol = r.get("symbol", "")
        sym_prices = price_map.setdefault(symbol, [])

        spread = float(r.get("spread", 0) or 0)
        slippage = float(r.get("slippage", 0) or 0)
        account_equity = float(r.get("equity", 0) or 0)
        margin_level = float(r.get("margin_level", 0) or 0)

        hour_sin = math.sin(2 * math.pi * t.hour / 24)
        hour_cos = math.cos(2 * math.pi * t.hour / 24)
        dow_sin = math.sin(2 * math.pi * t.weekday() / 7)
        dow_cos = math.cos(2 * math.pi * t.weekday() / 7)

        sl_dist = sl - price
        tp_dist = tp - price

        feat = {
            "symbol": symbol,
            "hour": t.hour,
            "day_of_week": t.weekday(),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "lots": lots,
            "profit": profit,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "spread": spread,
            "equity": account_equity,
            "margin_level": margin_level,
        }

        if use_slippage:
            feat["slippage"] = slippage

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

        if use_higher_timeframe:
            if use_sma:
                feat[f"sma_{higher_timeframe}"] = _sma(tf_prices, sma_window)
            if use_rsi:
                feat[f"rsi_{higher_timeframe}"] = _rsi(tf_prices, rsi_period)
            if use_macd:
                feat[f"macd_{higher_timeframe}"] = tf_macd
                feat[f"macd_signal_{higher_timeframe}"] = tf_macd_sig

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

        feature_dicts.append(feat)
        labels.append(label)
        sl_targets.append(sl_dist)
        tp_targets.append(tp_dist)
    return feature_dicts, np.array(labels), np.array(sl_targets), np.array(tp_targets)


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
    use_slippage: bool = False,
    use_volume: bool = False,
    volatility_series=None,
    use_higher_timeframe: bool = False,
    higher_timeframe: str = "H1",
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
):
    """Train a simple classifier model from the log directory."""

    cache_file = out_dir / "feature_cache.npz"

    existing_model = None
    if incremental:
        model_file = out_dir / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found for incremental training")
        with open(model_file) as f:
            existing_model = json.load(f)

    features = labels = sl_targets = tp_targets = None
    loaded_from_cache = False
    if cache_features and incremental and cache_file.exists() and existing_model is not None:
        try:
            cached = np.load(cache_file, allow_pickle=True)
            cached_names = list(cached.get("feature_names", []))
            if cached_names == existing_model.get("feature_names", []):
                features = [dict(x) for x in cached["feature_dicts"]]
                labels = cached["labels"]
                sl_targets = cached["sl_targets"]
                tp_targets = cached["tp_targets"]
                loaded_from_cache = True
        except Exception:
            features = None

    if features is None:
        rows_df = _load_logs(data_dir)
        encoder = None
        if encoder_file is not None and encoder_file.exists():
            with open(encoder_file) as f:
                encoder = json.load(f)
        features, labels, sl_targets, tp_targets = _extract_features(
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
            use_slippage=use_slippage,
            use_volume=use_volume,
            volatility=volatility_series,
            use_higher_timeframe=use_higher_timeframe,
            higher_timeframe=higher_timeframe,
            corr_pairs=corr_pairs,
            corr_window=corr_window,
            extra_price_series=extra_price_series,
            encoder=encoder,
        )
    else:
        encoder = None
        if encoder_file is not None and encoder_file.exists():
            with open(encoder_file) as f:
                encoder = json.load(f)

    if not features:
        raise ValueError(f"No training data found in {data_dir}")

    hidden_size = 8
    logreg_C = 1.0
    best_trial = None
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
    else:
        tscv = TimeSeriesSplit(n_splits=min(5, len(labels) - 1))
        # iterate through sequential splits and select the final one for validation
        for train_idx, val_idx in tscv.split(features):
            pass
        feat_train = [features[i] for i in train_idx]
        feat_val = [features[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        sl_train = sl_targets[train_idx]
        sl_val = sl_targets[val_idx]
        tp_train = tp_targets[train_idx]
        tp_val = tp_targets[val_idx]

        # if the training split ended up with only one class, fall back to using
        # all data for training so the model can be fit
        if len(np.unique(y_train)) < 2:
            feat_train, y_train = features, labels
            feat_val, y_val = [], np.array([])

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

    X_train_reg = vec.transform(feat_train_reg)
    X_val_reg = vec.transform(feat_val_reg) if feat_val_reg else np.empty((0, X_train_reg.shape[1]))

    if cache_features and not loaded_from_cache:
        feature_names_cache = vec.get_feature_names_out().tolist()
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            feature_dicts=np.array(features, dtype=object),
            labels=labels,
            sl_targets=sl_targets,
            tp_targets=tp_targets,
            feature_names=np.array(feature_names_cache),
        )

    if optuna_trials > 0 and HAS_OPTUNA:
        def _objective(trial):
            if model_type == "logreg":
                c = trial.suggest_float("C", 1e-3, 10.0, log=True)
                clf = LogisticRegression(max_iter=200, C=c)
            elif model_type == "xgboost":
                if XGBClassifier is None:
                    raise ImportError("xgboost is not installed")
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
            elif model_type == "lgbm":
                if LGBMClassifier is None:
                    raise ImportError("lightgbm is not installed")
                est = trial.suggest_int("n_estimators", 50, 300)
                lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                clf = LGBMClassifier(
                    n_estimators=est,
                    learning_rate=lr,
                    max_depth=depth,
                )
            elif model_type == "nn":
                h = trial.suggest_int("hidden_size", 4, 64)
                if HAS_TF:
                    clf = keras.Sequential([
                        keras.layers.Input(shape=(X_train.shape[1],)),
                        keras.layers.Dense(h, activation="relu"),
                        keras.layers.Dense(1, activation="sigmoid"),
                    ])
                    clf.compile(optimizer="adam", loss="binary_crossentropy")
                    clf.fit(X_train, y_train, epochs=50, verbose=0)
                else:
                    clf = MLPClassifier(hidden_layer_sizes=(h,), max_iter=500, random_state=42)
                    clf.fit(X_train, y_train)
            else:
                return 0.0

            if model_type == "nn" and HAS_TF:
                val_proba = clf.predict(X_val).reshape(-1) if len(y_val) > 0 else np.empty(0)
            else:
                val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)

            if len(y_val) > 0:
                t, _ = _best_threshold(y_val, val_proba)
                preds = (val_proba >= t).astype(int)
                return accuracy_score(y_val, preds)
            return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=optuna_trials)
        best_trial = study.best_trial
        if model_type == "logreg":
            logreg_C = float(best_trial.params["C"])
        elif model_type == "xgboost":
            n_estimators = int(best_trial.params["n_estimators"])
            learning_rate = float(best_trial.params["learning_rate"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "lgbm":
            n_estimators = int(best_trial.params["n_estimators"])
            learning_rate = float(best_trial.params["learning_rate"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "nn":
            hidden_size = int(best_trial.params["hidden_size"])

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        clf = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed")
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "nn":
        if HAS_TF:
            model_nn = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.Dense(hidden_size, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)] if early_stop else None
            model_nn.fit(
                X_train,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba = model_nn.predict(X_train).reshape(-1)
            val_proba = (
                model_nn.predict(X_val).reshape(-1) if len(y_val) > 0 else np.empty(0)
            )
            clf = model_nn
        else:
            clf = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            train_proba = clf.predict_proba(X_train)[:, 1]
            val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "lstm":
        if not HAS_TF:
            raise ImportError("TensorFlow is required for LSTM model")
        seq_len = sequence_length
        X_all = vec.fit_transform(features) if existing_model is None else vec.transform(features)
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
            X_val_seq, y_val = np.empty((0, seq_len, X_all.shape[1])), np.array([])
        else:
            X_train_seq, X_val_seq, y_train, y_val = train_test_split(
                X_all_seq,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
        model_nn = keras.Sequential([
            keras.layers.Input(shape=(seq_len, X_all.shape[1])),
            keras.layers.LSTM(8),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model_nn.compile(optimizer="adam", loss="binary_crossentropy")
        callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)] if early_stop else None
        model_nn.fit(
            X_train_seq,
            y_train,
            epochs=50,
            verbose=0,
            callbacks=callbacks,
        )
        train_proba = model_nn.predict(X_train_seq).reshape(-1)
        val_proba = model_nn.predict(X_val_seq).reshape(-1) if len(y_val) > 0 else np.empty(0)
        clf = model_nn
    elif model_type == "transformer":
        if not HAS_TF:
            raise ImportError("TensorFlow is required for transformer model")
        seq_len = sequence_length
        X_all = vec.fit_transform(features) if existing_model is None else vec.transform(features)
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
            X_val_seq, y_val = np.empty((0, seq_len, X_all.shape[1])), np.array([])
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
        callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)] if early_stop else None
        model_nn.fit(
            X_train_seq,
            y_train,
            epochs=50,
            verbose=0,
            callbacks=callbacks,
        )
        train_proba = model_nn.predict(X_train_seq).reshape(-1)
        val_proba = model_nn.predict(X_val_seq).reshape(-1) if len(y_val) > 0 else np.empty(0)
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
            clf = LogisticRegression(max_iter=200, C=logreg_C, warm_start=existing_model is not None)
            if existing_model is not None:
                clf.classes_ = np.array([0, 1])
                clf.coef_ = np.array([existing_model.get("coefficients", [])])
                clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
            clf.fit(X_train, y_train, sample_weight=sample_weight)
            train_proba = clf.predict_proba(X_train)[:, 1]
            val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)

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
        threshold, _ = _best_threshold(y_val, val_proba)
        val_preds = (val_proba >= threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_preds))
    else:
        threshold = 0.5
        val_acc = float("nan")
    train_preds = (train_proba >= threshold).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))

    hourly_thresholds = None
    if len(y_val) > 0:
        hours_val = np.array([f.get("hour", 0) for f in feat_val], dtype=int)
        hourly_thresholds = []
        for h in range(24):
            idx = np.where(hours_val == h)[0]
            if len(idx) > 0:
                t, _ = _best_threshold(y_val[idx], val_proba[idx])
            else:
                t = threshold
            hourly_thresholds.append(float(t))

    # statistics for feature scaling
    X_stats = vec.transform(feat_train)
    feature_mean = X_stats.mean(axis=0)
    feature_std = X_stats.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    # Compute SHAP feature importance on the training set
    try:
        import shap  # type: ignore

        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_train)
        importances = np.abs(shap_values.values).mean(axis=0)
        feature_importance = dict(
            zip(vec.get_feature_names_out().tolist(), importances.tolist())
        )
    except Exception:  # pragma: no cover - shap optional
        feature_importance = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "model_type": model_type,
        "weighted": sample_weight is not None,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "threshold": threshold,
        # main accuracy metric is validation performance when available
        "accuracy": val_acc,
        "num_samples": int(labels.shape[0]) + (int(existing_model.get("num_samples", 0)) if existing_model else 0),
        "feature_importance": feature_importance,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }
    if encoder is not None:
        model["encoder_weights"] = encoder.get("weights")
        model["encoder_window"] = encoder.get("window")
        if "centers" in encoder:
            model["encoder_centers"] = encoder.get("centers")
    if best_trial is not None:
        model["optuna_best_params"] = best_trial.params
        model["optuna_best_score"] = best_trial.value
    if hourly_thresholds is not None:
        model["hourly_thresholds"] = hourly_thresholds

    if model_type == "logreg":
        model["coefficients"] = clf.coef_[0].tolist()
        model["intercept"] = float(clf.intercept_[0])
    elif model_type == "xgboost":
        # approximate tree ensemble with linear model for MQL4 export
        logit_p = np.log(train_proba / (1.0 - train_proba + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].tolist()
        model["intercept"] = float(coef[0])

        # lookup probabilities per trading hour for simple export
        feature_names = vec.get_feature_names_out().tolist()
        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour" in f:
                f["hour"] = float(h)
            X_h = vec.transform([f])
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "lgbm":
        # approximate boosting model with linear regression for export
        logit_p = np.log(train_proba / (1.0 - train_proba + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].tolist()
        model["intercept"] = float(coef[0])

        feature_names = vec.get_feature_names_out().tolist()
        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour" in f:
                f["hour"] = float(h)
            X_h = vec.transform([f])
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "nn":
        if HAS_TF:
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
        model["sl_coefficients"] = sl_coef.tolist()
        model["sl_intercept"] = float(sl_inter)
        model["tp_coefficients"] = tp_coef.tolist()
        model["tp_intercept"] = float(tp_inter)

    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {out_dir / 'model.json'}")

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
    p.add_argument('--use-slippage', action='store_true', help='include slippage feature')
    p.add_argument('--use-higher-timeframe', action='store_true', help='include higher timeframe indicators')
    p.add_argument('--higher-timeframe', default='H1', help='timeframe for higher timeframe indicators')
    p.add_argument('--volatility-file', help='JSON file with precomputed volatility')
    p.add_argument('--grid-search', action='store_true', help='enable grid search with cross-validation')
    p.add_argument('--c-values', type=float, nargs='*')
    p.add_argument('--model-type', choices=['logreg', 'random_forest', 'xgboost', 'lgbm', 'nn', 'lstm', 'transformer'], default='logreg',
                   help='classifier type')
    p.add_argument('--sequence-length', type=int, default=5, help='sequence length for LSTM/transformer models')
    p.add_argument('--n-estimators', type=int, default=100, help='xgboost trees')
    p.add_argument('--learning-rate', type=float, default=0.1, help='xgboost learning rate')
    p.add_argument('--max-depth', type=int, default=3, help='xgboost tree depth')
    p.add_argument('--incremental', action='store_true', help='update existing model.json')
    p.add_argument('--cache-features', action='store_true', help='reuse cached feature matrix')
    p.add_argument('--corr-symbols', help='comma separated correlated symbol pairs e.g. EURUSD:USDCHF')
    p.add_argument('--corr-window', type=int, default=5, help='window for correlation calculations')
    p.add_argument('--optuna-trials', type=int, default=0, help='number of Optuna trials for hyperparameter search')
    p.add_argument('--encoder-file', help='JSON file with pretrained encoder weights')
    p.add_argument('--regress-sl-tp', action='store_true', help='learn SL/TP distance regressors')
    p.add_argument('--early-stop', action='store_true', help='enable early stopping for neural nets')
    args = p.parse_args()
    if args.volatility_file:
        import json
        with open(args.volatility_file) as f:
            vol_data = json.load(f)
    else:
        vol_data = None
    if args.corr_symbols:
        corr_pairs = [tuple(p.split(':')) for p in args.corr_symbols.split(',')]
    else:
        corr_pairs = None
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
        use_slippage=args.use_slippage,
        use_higher_timeframe=args.use_higher_timeframe,
        higher_timeframe=args.higher_timeframe,
        volatility_series=vol_data,
        grid_search=args.grid_search,
        c_values=args.c_values,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        incremental=args.incremental,
        sequence_length=args.sequence_length,
        corr_pairs=corr_pairs,
        corr_window=args.corr_window,
        optuna_trials=args.optuna_trials,
        regress_sl_tp=args.regress_sl_tp,
        early_stop=args.early_stop,
        encoder_file=Path(args.encoder_file) if args.encoder_file else None,
        cache_features=args.cache_features,
    )


if __name__ == '__main__':
    main()
