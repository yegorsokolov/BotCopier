"""Technical feature extraction utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd

from .registry import FEATURE_REGISTRY, register_feature

try:  # optional polars dependency
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False
from sklearn.linear_model import LinearRegression

from ..scripts.features import (
    KALMAN_DEFAULT_PARAMS,
    _atr,
    _bollinger,
    _is_doji,
    _is_engulfing,
    _is_hammer,
    _kalman_filter_series,
    _macd_update,
    _rsi,
    _sma,
)

try:  # Optional torch dependency
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # Optional graph dependency
    from ..scripts.graph_dataset import GraphDataset, compute_gnn_embeddings

    _HAS_TG = True
except ImportError:  # pragma: no cover - optional
    GraphDataset = None  # type: ignore
    compute_gnn_embeddings = None  # type: ignore
    _HAS_TG = False


logger = logging.getLogger(__name__)


@register_feature("technical")
def _extract_features_impl(
    df: pd.DataFrame | "pl.DataFrame",
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
    n_jobs: int | None = None,
) -> tuple[
    pd.DataFrame, list[str], dict[str, list[float]], dict[str, list[list[float]]]
]:
    """Attach graph embeddings, calendar flags and correlation features."""
    from . import engineering as fe  # avoid circular import

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
            except (OSError, ValueError) as exc:
                logger.exception("Failed to load graph dataset from %s", symbol_graph)
                g_dataset = None
    embeddings: dict[str, list[float]] = {}
    gnn_state: dict[str, list[list[float]]] = {}

    executor: ThreadPoolExecutor | None = None
    gnn_future = None
    if (
        n_jobs is not None
        and n_jobs > 1
        and g_dataset is not None
        and compute_gnn_embeddings is not None
    ):
        executor = ThreadPoolExecutor(max_workers=n_jobs)
        gnn_future = executor.submit(compute_gnn_embeddings, df.copy(), g_dataset)

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
        engulf_expr = (body > (prev_close - prev_open).abs()) & (
            ((pl.col("open") < prev_close) & (pl.col("close") > prev_open))
            | ((pl.col("open") > prev_close) & (pl.col("close") < prev_open))
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
            hammer = (
                (lower >= 2 * body)
                & (upper <= body)
                & (total > 0)
                & ((body / total) < 0.3)
            )
            doji = (body <= 0.1 * total) & (total != 0)
        prev_open = np.roll(opens, 1)
        prev_close = np.roll(closes, 1)
        prev_open[0] = np.nan
        prev_close[0] = np.nan
        engulf = (body > np.abs(prev_close - prev_open)) & (
            ((opens < prev_close) & (closes > prev_open))
            | ((opens > prev_close) & (closes < prev_open))
        )
        df["pattern_hammer"] = np.where(np.isnan(hammer), 0.0, hammer.astype(float))
        df["pattern_doji"] = np.where(np.isnan(doji), 0.0, doji.astype(float))
        df["pattern_engulfing"] = np.where(
            np.isnan(prev_open) | np.isnan(prev_close), 0.0, engulf.astype(float)
        )
        feature_names.extend(["pattern_hammer", "pattern_doji", "pattern_engulfing"])

    if fe._CONFIG.is_enabled("orderbook") and all(
        c in df.columns for c in ["bid_depth", "ask_depth", "bid", "ask"]
    ):
        bid_depth = df["bid_depth"].apply(lambda x: np.asarray(x, dtype=float))
        ask_depth = df["ask_depth"].apply(lambda x: np.asarray(x, dtype=float))

        bid_top = bid_depth.apply(lambda a: float(a[0]) if a.size else 0.0)
        ask_top = ask_depth.apply(lambda a: float(a[0]) if a.size else 0.0)
        bid_vol = bid_depth.apply(lambda a: float(np.nansum(a)))
        ask_vol = ask_depth.apply(lambda a: float(np.nansum(a)))
        total_vol = bid_vol + ask_vol + 1e-9

        bids = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
        asks = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
        df["depth_microprice"] = (asks * bid_top + bids * ask_top) / (
            bid_top + ask_top + 1e-9
        )
        df["depth_vol_imbalance"] = (bid_vol - ask_vol) / total_vol
        prev_bid_vol = bid_vol.shift(1).fillna(bid_vol.iloc[0])
        prev_ask_vol = ask_vol.shift(1).fillna(ask_vol.iloc[0])
        df["depth_order_flow_imbalance"] = (bid_vol - prev_bid_vol) - (
            ask_vol - prev_ask_vol
        )
        feature_names.extend(
            [
                "depth_microprice",
                "depth_vol_imbalance",
                "depth_order_flow_imbalance",
            ]
        )

    if fe._CONFIG.is_enabled("kalman"):
        params = fe._CONFIG.kalman_params
        p_var = params.get("process_var", KALMAN_DEFAULT_PARAMS["process_var"])
        m_var = params.get("measurement_var", KALMAN_DEFAULT_PARAMS["measurement_var"])
        if "close" in df.columns:
            closes = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(float)
            lvl_vals, tr_vals = _kalman_filter_series(closes, p_var, m_var)
            df["kalman_price_level"] = lvl_vals
            df["kalman_price_trend"] = tr_vals
            feature_names.extend(["kalman_price_level", "kalman_price_trend"])
        if "volume" in df.columns:
            vols = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(float)
            lvl_vals, tr_vals = _kalman_filter_series(vols, p_var, m_var)
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
            except (OSError, RuntimeError, ValueError) as exc:
                logger.exception("Failed to load tick encoder from %s", tick_encoder)

    if neighbor_corr_windows is not None and len(neighbor_corr_windows) > 0:
        if "symbol" in df.columns and "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            corr_cols: list[str] = []
            for win in neighbor_corr_windows:
                col = f"neighbor_corr_{win}"
                corr_cols.append(col)
                df[col] = (
                    prices.groupby(df["symbol"])
                    .pct_change()
                    .rolling(win)
                    .corr()
                    .fillna(0.0)
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

    if gnn_future is not None:
        try:
            embeddings, gnn_state = gnn_future.result()
        except (RuntimeError, ValueError) as exc:
            logger.exception("Failed to compute GNN embeddings")
            embeddings, gnn_state = {}, {}
        finally:
            executor.shutdown()
    elif g_dataset is not None and compute_gnn_embeddings is not None:
        try:
            embeddings, gnn_state = compute_gnn_embeddings(df, g_dataset)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Failed to compute GNN embeddings")
            embeddings, gnn_state = {}, {}
    if embeddings and "symbol" in df.columns:
        emb_dim = len(next(iter(embeddings.values())))
        sym_series = df["symbol"].astype(str)
        for i in range(emb_dim):
            col = f"graph_emb{i}"
            df[col] = sym_series.map(
                lambda s: embeddings.get(s, [0.0] * emb_dim)[i]
            )
        feature_names = feature_names + [f"graph_emb{i}" for i in range(emb_dim)]

    return df, feature_names, embeddings, gnn_state


def _extract_features(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    **kwargs,
) -> tuple[
    pd.DataFrame | "pl.DataFrame",
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]:
    """Run enabled feature plugins and return augmented ``df`` and metadata."""
    from .engineering import _CONFIG, _FEATURE_RESULTS  # late import to avoid circular deps

    key = id(df)
    if key in _FEATURE_RESULTS:
        logger.info("cache hit for _extract_features")
        return _FEATURE_RESULTS[key]  # type: ignore[return-value]

    plugins = ["technical"] + [
        name
        for name in _CONFIG.enabled_features
        if name != "technical" and name in FEATURE_REGISTRY
    ]
    embeddings: dict[str, list[float]] = {}
    gnn_state: dict[str, list[list[float]]] = {}
    for name in plugins:
        func = FEATURE_REGISTRY.get(name)
        if func is None:
            logger.warning("Feature plugin %s not found", name)
            continue
        df, feature_names, emb, gnn = func(df, feature_names, **kwargs)
        embeddings.update(emb or {})
        gnn_state.update(gnn or {})
    _FEATURE_RESULTS[key] = (df, feature_names, embeddings, gnn_state)
    return df, feature_names, embeddings, gnn_state


@register_feature("neutralize")
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
        logging.info(
            "Neutralised %s: variance %.6f -> %.6f", col, var_before, var_after
        )
        df[col] = resid
    return df, feature_names
