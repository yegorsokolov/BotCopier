"""Feature engineering utilities for BotCopier."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Sequence

import logging
import math

from functools import wraps

import json

import numpy as np
import pandas as pd
try:  # optional polars dependency
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from joblib import Memory

from scripts.features import _is_hammer, _is_doji, _is_engulfing
from scripts.features import (
    _sma,
    _rsi,
    _bollinger,
    _macd_update,
    _atr,
    _kalman_update,
    KALMAN_DEFAULT_PARAMS,
)

try:  # Optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # Optional graph dependency
    from scripts.graph_dataset import GraphDataset, compute_gnn_embeddings

    _HAS_TG = True
except Exception:  # pragma: no cover - optional
    GraphDataset = None  # type: ignore
    compute_gnn_embeddings = None  # type: ignore
    _HAS_TG = False

# Expose Kalman configuration so callers can toggle via package
USE_KALMAN_FEATURES: bool = False
KALMAN_PARAMS: dict = KALMAN_DEFAULT_PARAMS.copy()

# global cache configuration
_MEMORY = Memory(None, verbose=0)


def configure_cache(cache_dir: Path | str | None) -> None:
    """Configure joblib cache directory for expensive feature functions."""
    global _MEMORY, _augment_dataframe, _augment_dtw_dataframe, _extract_features
    _MEMORY = Memory(str(cache_dir) if cache_dir else None, verbose=0)

    # rewrap cached functions whenever cache location changes
    _augment_dataframe = _cache_with_logging(_augment_dataframe_impl, "_augment_dataframe")
    _augment_dtw_dataframe = _cache_with_logging(
        _augment_dtw_dataframe_impl, "_augment_dtw_dataframe"
    )
    _extract_features = _cache_with_logging(_extract_features_impl, "_extract_features")


def clear_cache() -> None:
    """Remove all cached feature computations."""
    _MEMORY.clear()


def _cache_with_logging(func, name: str):
    cached_func = _MEMORY.cache(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if cached_func.check_call_in_cache(*args, **kwargs):
            logging.info("cache hit for %s", name)
        return cached_func(*args, **kwargs)

    wrapper.clear_cache = cached_func.clear  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Feature clipping and anomaly scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------


def _augment_dataframe_impl(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Return DataFrame with additional augmented rows using mixup and jitter."""
    if ratio <= 0 or df.empty:
        return df

    n = len(df)
    n_aug = max(1, int(n * ratio))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[num_cols].std().replace(0, 1).to_numpy()

    aug_rows: list[pd.Series] = []
    for _ in range(n_aug):
        i1, i2 = np.random.randint(0, n, size=2)
        lam = np.random.beta(0.4, 0.4)
        row1 = df.iloc[i1]
        row2 = df.iloc[i2]
        new_row = row1.copy()
        if num_cols:
            mix = lam * row1[num_cols].to_numpy(dtype=float) + (1 - lam) * row2[num_cols].to_numpy(dtype=float)
            jitter = np.random.normal(0.0, 0.01, size=len(num_cols)) * stats
            new_row[num_cols] = mix + jitter
        if "event_time" in df.columns and pd.notnull(new_row.get("event_time")):
            delta = np.random.uniform(-60, 60)
            new_row["event_time"] = new_row["event_time"] + pd.to_timedelta(delta, unit="s")
        aug_rows.append(new_row)

    aug_df = pd.DataFrame(aug_rows)
    logging.info("Augmenting data with %d synthetic rows (ratio %.3f)", n_aug, n_aug / n)
    return pd.concat([df, aug_df], ignore_index=True)


def _dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[list[tuple[int, int]], float]:
    """Return optimal DTW alignment path between two sequences and its cost."""
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path, float(dp[n, m])


def _augment_dtw_dataframe_impl(df: pd.DataFrame, ratio: float, window: int = 3) -> pd.DataFrame:
    """Return DataFrame augmented by DTW-based sequence mixup."""
    if ratio <= 0 or len(df) < 2:
        return df

    n = len(df)
    n_aug = max(1, int(n * ratio))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df

    windows: list[np.ndarray] = []
    for start in range(0, n - window + 1):
        windows.append(df.iloc[start : start + window][num_cols].to_numpy(dtype=float))
    if not windows:
        windows.append(df[num_cols].to_numpy(dtype=float))
        window = len(df)

    aug_rows: list[pd.Series] = []
    while len(aug_rows) < n_aug:
        i1 = np.random.randint(0, len(windows))
        seq1 = windows[i1]
        best_j = None
        best_dist = np.inf
        for j in range(len(windows)):
            if j == i1:
                continue
            _, dist = _dtw_path(seq1, windows[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None:
            break
        seq2 = windows[best_j]
        path, _ = _dtw_path(seq1, seq2)
        lam = np.random.beta(0.4, 0.4)
        for idx1, idx2 in path:
            row1 = df.iloc[i1 + idx1]
            row2 = df.iloc[best_j + idx2]
            new_row = row1.copy()
            new_row[num_cols] = lam * row1[num_cols].to_numpy(dtype=float) + (1 - lam) * row2[num_cols].to_numpy(dtype=float)
            if "event_time" in df.columns:
                t1 = pd.to_datetime(row1.get("event_time"))
                t2 = pd.to_datetime(row2.get("event_time"))
                if pd.notnull(t1) and pd.notnull(t2):
                    new_row["event_time"] = t1 + (t2 - t1) * (1 - lam)
            new_row["aug_ratio"] = lam
            aug_rows.append(new_row)
            if len(aug_rows) >= n_aug:
                break

    if not aug_rows:
        return df
    aug_df = pd.DataFrame(aug_rows)
    logging.info("DTW augmenting data with %d synthetic rows (ratio %.3f)", len(aug_df), len(aug_df) / n)
    return pd.concat([df, aug_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def _extract_features_impl(
    df: pd.DataFrame | 'pl.DataFrame',
    feature_names: list[str],
    *,
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    event_window: float = 60.0,
    news_sentiment: pd.DataFrame | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    regime_model: dict | str | Path | None = None,
    tick_encoder: Path | None = None,
    calendar_features: bool = True,
    pca_components: dict | None = None,
    rank_features: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, list[float]], dict[str, list[list[float]]]]:
    """Attach graph embeddings, calendar flags and correlation features.

    The function accepts either a :class:`pandas.DataFrame` or, when the
    optional ``polars`` dependency is installed, a :class:`polars.DataFrame`.
    Polars inputs are processed using vectorised expressions and converted
    back to pandas for downstream compatibility.
    """
    g_dataset: GraphDataset | None = None
    graph_data: dict | None = None
    if symbol_graph is not None:
        if not isinstance(symbol_graph, dict):
            with open(symbol_graph) as f_sg:
                graph_data = json.load(f_sg)
        else:
            graph_data = symbol_graph
        if _HAS_TG and not isinstance(symbol_graph, dict):
            try:
                g_dataset = GraphDataset(symbol_graph)
            except Exception:
                g_dataset = None
    embeddings: dict[str, list[float]] = {}
    gnn_state: dict[str, list[list[float]]] = {}

    use_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)
    if use_polars and "event_time" in df.columns:
        df = df.with_columns(pl.col("event_time").cast(pl.Datetime("ns")))
        if calendar_features:
            df = df.with_columns(
                [
                    pl.col("event_time").dt.hour().alias("hour"),
                    pl.col("event_time").dt.weekday().alias("dayofweek"),
                    pl.col("event_time").dt.month().alias("month"),
                ]
            )
            df = df.with_columns(
                [
                    (2 * np.pi * pl.col("hour") / 24.0).sin().alias("hour_sin"),
                    (2 * np.pi * pl.col("hour") / 24.0).cos().alias("hour_cos"),
                    (2 * np.pi * pl.col("dayofweek") / 7.0).sin().alias("dow_sin"),
                    (2 * np.pi * pl.col("dayofweek") / 7.0).cos().alias("dow_cos"),
                    (2 * np.pi * (pl.col("month") - 1) / 12.0).sin().alias("month_sin"),
                    (2 * np.pi * (pl.col("month") - 1) / 12.0).cos().alias("month_cos"),
                ]
            )
            for col in [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
            ]:
                if col not in feature_names:
                    feature_names.append(col)
    elif "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        if calendar_features:
            if "hour" not in df.columns:
                df["hour"] = df["event_time"].dt.hour.astype(int)
            if "dayofweek" not in df.columns:
                df["dayofweek"] = df["event_time"].dt.dayofweek.astype(int)
            if "month" not in df.columns:
                df["month"] = df["event_time"].dt.month.astype(int)
            if "hour_sin" not in df.columns:
                df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
                df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
            if "dow_sin" not in df.columns:
                df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
                df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
            if "month_sin" not in df.columns:
                df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
                df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
            for col in [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
            ]:
                if col in df.columns and col not in feature_names:
                    feature_names.append(col)

    if use_polars and all(c in df.columns for c in ["open", "high", "low", "close"]):
        body = (pl.col("close") - pl.col("open")).abs()
        upper = pl.col("high") - pl.max_horizontal("open", "close")
        lower = pl.min_horizontal("open", "close") - pl.col("low")
        total = pl.col("high") - pl.col("low")
        hammer_expr = (
            (lower >= 2 * body) & (upper <= body) & (total > 0) & ((body / total) < 0.3)
        )
        doji_expr = (body <= 0.1 * total) & (total != 0)
        prev_open = pl.col("open").shift(1)
        prev_close = pl.col("close").shift(1)
        engulf_expr = (
            (body > (prev_close - prev_open).abs())
            & (((pl.col("open") < prev_close) & (pl.col("close") > prev_open))
               | ((pl.col("open") > prev_close) & (pl.col("close") < prev_open)))
        )
        df = df.with_columns(
            [
                hammer_expr.cast(pl.Float64).fill_null(0.0).alias("pattern_hammer"),
                doji_expr.cast(pl.Float64).fill_null(0.0).alias("pattern_doji"),
                engulf_expr.cast(pl.Float64).fill_null(0.0).alias("pattern_engulfing"),
            ]
        )
        feature_names.extend(["pattern_hammer", "pattern_doji", "pattern_engulfing"])
        df = df.to_pandas()
        use_polars = False
    elif all(c in df.columns for c in ["open", "high", "low", "close"]):
        opens = pd.to_numeric(df["open"], errors="coerce").to_numpy()
        highs = pd.to_numeric(df["high"], errors="coerce").to_numpy()
        lows = pd.to_numeric(df["low"], errors="coerce").to_numpy()
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy()
        body = np.abs(closes - opens)
        upper = highs - np.maximum(opens, closes)
        lower = np.minimum(opens, closes) - lows
        total = highs - lows
        with np.errstate(divide="ignore", invalid="ignore"):
            hammer = (lower >= 2 * body) & (upper <= body) & (total > 0) & ((body / total) < 0.3)
            doji = (body <= 0.1 * total) & (total != 0)
        prev_open = np.roll(opens, 1)
        prev_close = np.roll(closes, 1)
        prev_open[0] = np.nan
        prev_close[0] = np.nan
        engulf = (
            (body > np.abs(prev_close - prev_open))
            & (
                ((opens < prev_close) & (closes > prev_open))
                | ((opens > prev_close) & (closes < prev_open))
            )
        )
        df["pattern_hammer"] = np.where(np.isnan(hammer), 0.0, hammer.astype(float))
        df["pattern_doji"] = np.where(np.isnan(doji), 0.0, doji.astype(float))
        df["pattern_engulfing"] = np.where(
            np.isnan(prev_open) | np.isnan(prev_close), 0.0, engulf.astype(float)
        )
        feature_names.extend(["pattern_hammer", "pattern_doji", "pattern_engulfing"])

    if USE_KALMAN_FEATURES:
        params = KALMAN_PARAMS
        if "close" in df.columns:
            price_state: dict = {}
            lvl_vals: list[float] = []
            tr_vals: list[float] = []
            for p in pd.to_numeric(df["close"], errors="coerce").fillna(0.0):
                lvl, tr = _kalman_update(price_state, float(p), **params)
                lvl_vals.append(lvl)
                tr_vals.append(tr)
            df["kalman_price_level"] = lvl_vals
            df["kalman_price_trend"] = tr_vals
            feature_names.extend(["kalman_price_level", "kalman_price_trend"])
        if "volume" in df.columns:
            vol_state: dict = {}
            lvl_vals: list[float] = []
            tr_vals: list[float] = []
            for v in pd.to_numeric(df["volume"], errors="coerce").fillna(0.0):
                lvl, tr = _kalman_update(vol_state, float(v), **params)
                lvl_vals.append(lvl)
                tr_vals.append(tr)
            df["kalman_volume_level"] = lvl_vals
            df["kalman_volume_trend"] = tr_vals
            feature_names.extend(["kalman_volume_level", "kalman_volume_trend"])

    if tick_encoder is not None and _HAS_TORCH:
        tick_cols = [c for c in df.columns if c.startswith("tick_")]
        if tick_cols:
            tick_cols = sorted(
                tick_cols,
                key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0,
            )
            try:
                state = torch.load(tick_encoder, map_location="cpu")
                weight = state.get("state_dict", {}).get("weight")
                if weight is not None:
                    weight_t = weight.float().t()
                    window = weight_t.shape[0]
                    use_cols = tick_cols[:window]
                    X = df[use_cols].to_numpy(dtype=float)
                    df["tick_emb"] = X @ weight_t.to("cpu").numpy()
                    feature_names.append("tick_emb")
            except Exception:
                pass

    if neighbor_corr_windows is not None and len(neighbor_corr_windows) > 0:
        if "symbol" in df.columns and "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            corr_cols: list[str] = []
            for win in neighbor_corr_windows:
                col = f"neighbor_corr_{win}"
                corr_cols.append(col)
                df[col] = (
                    prices.groupby(df["symbol"]).pct_change().rolling(win).corr().fillna(0.0)
                )
            feature_names.extend(corr_cols)

    if news_sentiment is not None and "symbol" in df.columns:
        df = df.merge(news_sentiment, on="symbol", how="left")
        for col in news_sentiment.columns:
            if col != "symbol" and col not in feature_names:
                feature_names.append(col)

    if rank_features and "symbol" in df.columns:
        idx_col = "event_time" if "event_time" in df.columns else None
        group = df[idx_col] if idx_col else df.index
        if "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            returns = prices.groupby(df["symbol"]).pct_change().fillna(0.0)
            r_rank = returns.groupby(group).rank(method="min") - 1
            r_count = returns.groupby(group).transform("count") - 1
            df["ret_rank"] = np.where(r_count > 0, r_rank / r_count, 0.5)
            if "ret_rank" not in feature_names:
                feature_names.append("ret_rank")
        if "volume" in df.columns:
            vols = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            v_rank = vols.groupby(group).rank(method="min") - 1
            v_count = vols.groupby(group).transform("count") - 1
            df["vol_rank"] = np.where(v_count > 0, v_rank / v_count, 0.5)
            if "vol_rank" not in feature_names:
                feature_names.append("vol_rank")

    if g_dataset is not None and compute_gnn_embeddings is not None:
        try:
            embeddings, gnn_state = compute_gnn_embeddings(df, g_dataset)
        except Exception:
            embeddings, gnn_state = {}, {}
        if embeddings and "symbol" in df.columns:
            emb_dim = len(next(iter(embeddings.values())))
            sym_series = df["symbol"].astype(str)
            for i in range(emb_dim):
                col = f"graph_emb{i}"
                df[col] = sym_series.map(lambda s: embeddings.get(s, [0.0] * emb_dim)[i])
            feature_names = feature_names + [f"graph_emb{i}" for i in range(emb_dim)]

    return df, feature_names, embeddings, gnn_state


def _neutralize_against_market_index(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Neutralise features by removing linear dependence on market index."""
    if "symbol" not in df.columns:
        return df, feature_names
    price_col = next((c for c in ["price", "bid", "ask"] if c in df.columns), None)
    if price_col is None:
        return df, feature_names

    prices = pd.to_numeric(df[price_col], errors="coerce")
    returns = prices.groupby(df["symbol"]).pct_change().fillna(0.0)
    mkt = returns.groupby(df.index).transform("mean")
    if np.var(mkt) == 0:
        return df, feature_names
    X = mkt.to_numpy().reshape(-1, 1)

    for col in feature_names:
        y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        if np.var(y) == 0:
            continue
        lr = LinearRegression().fit(X, y)
        pred = lr.predict(X)
        resid = y - pred
        var_before = float(np.var(y))
        var_after = float(np.var(resid))
        logging.info("Neutralised %s: variance %.6f -> %.6f", col, var_before, var_after)
        df[col] = resid
    return df, feature_names
  
configure_cache(None)


def train(*args, **kwargs):
    from botcopier.training.pipeline import train as _train

    return _train(*args, **kwargs)
