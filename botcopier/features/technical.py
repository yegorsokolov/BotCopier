"""Technical feature extraction utilities."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import signal

try:  # optional dask support
    import dask.dataframe as dd  # type: ignore

    _HAS_DASK = True
except Exception:  # pragma: no cover - optional
    dd = None  # type: ignore
    _HAS_DASK = False

from .plugins import FEATURE_REGISTRY, register_feature, load_plugins

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
    _fractal_dimension,
    _hurst_exponent,
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

try:  # optional networkx dependency
    import networkx as nx  # type: ignore

    _HAS_NX = True
except Exception:  # pragma: no cover - optional
    nx = None  # type: ignore
    _HAS_NX = False


logger = logging.getLogger(__name__)


_DEPTH_CNN_STATE: dict | None = None
_GRAPH_SNAPSHOT: dict | None = None

# Lookback windows for fractal metrics
HURST_WINDOW = 50
FRACTAL_DIM_WINDOW = 50

# Parameters for cross-spectral density features
CSD_WINDOW = 16
CSD_FREQ_BINS = 16
_CSD_PARAMS: dict | None = None

_SYMBOLIC_CACHE: dict | None = None


def _load_symbolic_indicators(model_json: Path | str | None) -> dict:
    path = Path(model_json or "model.json")
    global _SYMBOLIC_CACHE
    if _SYMBOLIC_CACHE is None or _SYMBOLIC_CACHE.get("_path") != path:
        try:
            data = json.loads(path.read_text())
            inds = data.get("symbolic_indicators", {})
        except Exception:
            inds = {}
        _SYMBOLIC_CACHE = {"_path": path, "data": inds}
    return _SYMBOLIC_CACHE.get("data", {})


_SYMBOLIC_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "neg": np.negative,
    "max": np.maximum,
    "min": np.minimum,
}


def _apply_symbolic_indicators(
    df: pd.DataFrame, feature_names: list[str], model_json: Path | str | None
) -> tuple[pd.DataFrame, list[str]]:
    sym = _load_symbolic_indicators(model_json)
    formulas = sym.get("formulas") or []
    base_feats = sym.get("feature_names") or []
    if not formulas or not base_feats:
        return df, feature_names
    env = {
        name: pd.to_numeric(
            df.get(name, pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        for name in base_feats
    }
    env.update(_SYMBOLIC_FUNCS)
    for idx, formula in enumerate(formulas):
        col = f"sym_{idx}"
        try:
            df[col] = eval(formula, {"__builtins__": {}}, env)
            if col not in feature_names:
                feature_names.append(col)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to evaluate symbolic indicator %s", formula)
    return df, feature_names


def _compute_entity_graph_features(
    df: pd.DataFrame,
    feature_names: list[str],
    graph: "nx.Graph",
) -> None:
    """Compute simple graph-derived features for each symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a ``symbol`` column.
    feature_names : list[str]
        List of feature names to append to.
    graph : nx.Graph
        Graph linking symbols, articles and entities.

    Adds two columns to ``df``:

    ``graph_article_count``
        Number of articles reachable from the symbol through entity/sector
        relationships.

    ``graph_sentiment``
        Mean sentiment across those articles.  Missing sentiments default to
        ``0.0``.
    """

    global _GRAPH_SNAPSHOT

    symbols = df.get("symbol")
    if symbols is None or symbols.empty:
        return

    if not _HAS_NX:
        return

    counts: list[int] = []
    sentiments: list[float] = []
    for sym in symbols.astype(str):
        count = 0
        sent_vals: list[float] = []
        if graph.has_node(sym):
            entities = [
                n
                for n in graph.neighbors(sym)
                if graph.nodes[n].get("type") in {"entity", "company"}
            ]
            for ent in entities:
                articles = [
                    a
                    for a in graph.neighbors(ent)
                    if graph.nodes[a].get("type") == "article"
                ]
                count += len(articles)
                sent_vals.extend(graph.nodes[a].get("sentiment", 0.0) for a in articles)
                sectors = [
                    s
                    for s in graph.neighbors(ent)
                    if graph.nodes[s].get("type") == "sector"
                ]
                for sec in sectors:
                    peers = [
                        p
                        for p in graph.neighbors(sec)
                        if graph.nodes[p].get("type") in {"entity", "company"}
                    ]
                    for peer in peers:
                        if peer == ent:
                            continue
                        peer_articles = [
                            a
                            for a in graph.neighbors(peer)
                            if graph.nodes[a].get("type") == "article"
                        ]
                        count += len(peer_articles)
                        sent_vals.extend(
                            graph.nodes[a].get("sentiment", 0.0) for a in peer_articles
                        )
        counts.append(count)
        sentiments.append(sum(sent_vals) / len(sent_vals) if sent_vals else 0.0)

    df["graph_article_count"] = counts
    df["graph_sentiment"] = sentiments
    for col in ["graph_article_count", "graph_sentiment"]:
        if col not in feature_names:
            feature_names.append(col)

    try:
        _GRAPH_SNAPSHOT = nx.node_link_data(graph)
    except Exception:  # pragma: no cover - best effort
        _GRAPH_SNAPSHOT = None


if _HAS_TORCH:

    class _DepthCNN(torch.nn.Module):
        """Tiny CNN used to embed orderbook depth snapshots."""

        def __init__(self, state: dict | None = None) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(2, 3))
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=(1, 1))
            if state:
                self._load(state)

        def _load(self, state: dict) -> None:
            self.conv1.weight.data = torch.tensor(state["conv1_weight"]).reshape_as(
                self.conv1.weight
            )
            self.conv1.bias.data = torch.tensor(state["conv1_bias"])
            self.conv2.weight.data = torch.tensor(state["conv2_weight"]).reshape_as(
                self.conv2.weight
            )
            self.conv2.bias.data = torch.tensor(state["conv2_bias"])

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            return x.view(x.size(0), -1)


def _depth_cnn_features(
    bids: pd.Series,
    asks: pd.Series,
    state: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute CNN embeddings for bid/ask depth arrays."""
    if not _HAS_TORCH:
        return np.zeros((len(bids), 0)), {}

    if state is None:
        flat_vals = [b.ravel() for b in bids if b.size] + [
            a.ravel() for a in asks if a.size
        ]
        all_vals = np.concatenate(flat_vals) if flat_vals else np.array([0.0])
        mean = float(all_vals.mean())
        std = float(all_vals.std() + 1e-6)
        torch.manual_seed(0)
        model = _DepthCNN()
        state = {
            "conv1_weight": model.conv1.weight.detach().cpu().numpy().tolist(),
            "conv1_bias": model.conv1.bias.detach().cpu().numpy().tolist(),
            "conv2_weight": model.conv2.weight.detach().cpu().numpy().tolist(),
            "conv2_bias": model.conv2.bias.detach().cpu().numpy().tolist(),
            "mean": mean,
            "std": std,
        }
    else:
        mean = float(state.get("mean", 0.0))
        std = float(state.get("std", 1.0))
        model = _DepthCNN(state)

    model.eval()
    emb_list: list[list[float]] = []
    for b, a in zip(bids, asks):
        if b.size == 0 or a.size == 0:
            emb_list.append([0.0] * 4)
            continue
        depth = np.stack([b, a], axis=0)
        depth = (depth - mean) / (std + 1e-9)
        t = torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(t).squeeze(0).cpu().numpy()
        emb_list.append(out.tolist())
    return np.asarray(emb_list, dtype=float), state


@register_feature("technical")
def _extract_features_impl(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    event_window: float = 60.0,
    news_sentiment: pd.DataFrame | None = None,
    entity_graph: "nx.Graph" | dict | str | Path | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    regime_model: dict | str | Path | None = None,
    tick_encoder: Path | None = None,
    depth_cnn: dict | None = None,
    calendar_features: bool = True,
    pca_components: dict | None = None,
    rank_features: bool = False,
    n_jobs: int | None = None,
    model_json: Path | str | None = None,
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

        if _HAS_TORCH:
            global _DEPTH_CNN_STATE
            emb, _DEPTH_CNN_STATE = _depth_cnn_features(
                bid_depth, ask_depth, depth_cnn or _DEPTH_CNN_STATE
            )
            for i in range(emb.shape[1]):
                col = f"depth_cnn_{i}"
                df[col] = emb[:, i]
                if col not in feature_names:
                    feature_names.append(col)

    if fe._CONFIG.is_enabled("kalman"):
        params = fe._CONFIG.kalman_params
        p_var = params.get("process_var", KALMAN_DEFAULT_PARAMS["process_var"])
        m_var = params.get("measurement_var", KALMAN_DEFAULT_PARAMS["measurement_var"])
        if "close" in df.columns:
            closes = (
                pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(float)
            )
            lvl_vals, tr_vals = _kalman_filter_series(closes, p_var, m_var)
            df["kalman_price_level"] = lvl_vals
            df["kalman_price_trend"] = tr_vals
            feature_names.extend(["kalman_price_level", "kalman_price_trend"])
        if "volume" in df.columns:
            vols = (
                pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(float)
            )
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

    if "symbol" in df.columns and "price" in df.columns:
        prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        hurst_vals = pd.Series(0.5, index=df.index, dtype=float)
        frac_vals = pd.Series(1.5, index=df.index, dtype=float)
        for sym, group in df.groupby("symbol"):
            p = prices.loc[group.index].to_list()
            h_list: list[float] = []
            f_list: list[float] = []
            for i in range(len(p)):
                window_p = p[max(0, i - HURST_WINDOW + 1) : i + 1]
                h = _hurst_exponent(window_p, len(window_p))
                h_list.append(h)
                f_list.append(_fractal_dimension(window_p, len(window_p)))
            hurst_vals.loc[group.index] = h_list
            frac_vals.loc[group.index] = f_list
        df["hurst"] = hurst_vals.to_numpy()
        df["fractal_dim"] = frac_vals.to_numpy()
        feature_names.extend(["hurst", "fractal_dim"])

    if (
        fe._CONFIG.is_enabled("csd")
        and graph_data
        and "symbol" in df.columns
        and "price" in df.columns
    ):
        symbols = graph_data.get("symbols", [])
        edge_index = graph_data.get("edge_index", [])
        pairs: set[tuple[str, str]] = set()
        if isinstance(edge_index, list) and len(edge_index) == 2:
            for a, b in zip(edge_index[0], edge_index[1]):
                if a < len(symbols) and b < len(symbols):
                    sa, sb = symbols[a], symbols[b]
                    if sa != sb:
                        pairs.add(tuple(sorted((sa, sb))))
        idx_col = "event_time" if "event_time" in df.columns else None
        idx = df[idx_col] if idx_col else df.index
        price_wide = (
            df.assign(_idx=idx)
            .pivot(index="_idx", columns="symbol", values="price")
            .sort_index()
            .ffill()
            .fillna(0.0)
        )
        global _CSD_PARAMS
        _CSD_PARAMS = {"window": CSD_WINDOW, "freq_bins": CSD_FREQ_BINS}
        for a, b in pairs:
            if a not in price_wide.columns or b not in price_wide.columns:
                continue
            sa = price_wide[a].to_numpy()
            sb = price_wide[b].to_numpy()
            freq_arr = np.zeros(len(price_wide))
            coh_arr = np.zeros(len(price_wide))
            for i in range(CSD_WINDOW - 1, len(price_wide)):
                xa = sa[i - CSD_WINDOW + 1 : i + 1]
                xb = sb[i - CSD_WINDOW + 1 : i + 1]
                f, Pxy = signal.csd(
                    xa, xb, nperseg=CSD_WINDOW, nfft=CSD_FREQ_BINS, scaling="spectrum"
                )
                _, Pxx = signal.csd(
                    xa, xa, nperseg=CSD_WINDOW, nfft=CSD_FREQ_BINS, scaling="spectrum"
                )
                _, Pyy = signal.csd(
                    xb, xb, nperseg=CSD_WINDOW, nfft=CSD_FREQ_BINS, scaling="spectrum"
                )
                coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-9)
                idx_max = int(np.argmax(coh))
                freq_arr[i] = float(f[idx_max])
                coh_arr[i] = float(np.real(coh[idx_max]))
            freq_series = pd.Series(freq_arr, index=price_wide.index)
            coh_series = pd.Series(coh_arr, index=price_wide.index)
            col_freq_ab = f"csd_freq_{b}"
            col_coh_ab = f"csd_coh_{b}"
            col_freq_ba = f"csd_freq_{a}"
            col_coh_ba = f"csd_coh_{a}"
            mask_a = df["symbol"] == a
            mask_b = df["symbol"] == b
            times_a = df.loc[mask_a, idx_col] if idx_col else df.loc[mask_a].index
            times_b = df.loc[mask_b, idx_col] if idx_col else df.loc[mask_b].index
            df.loc[mask_a, col_freq_ab] = (
                freq_series.reindex(times_a).fillna(0.0).to_numpy()
            )
            df.loc[mask_a, col_coh_ab] = (
                coh_series.reindex(times_a).fillna(0.0).to_numpy()
            )
            df.loc[mask_b, col_freq_ba] = (
                freq_series.reindex(times_b).fillna(0.0).to_numpy()
            )
            df.loc[mask_b, col_coh_ba] = (
                coh_series.reindex(times_b).fillna(0.0).to_numpy()
            )
            for col in [col_freq_ab, col_coh_ab, col_freq_ba, col_coh_ba]:
                if col not in feature_names:
                    feature_names.append(col)

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

    if entity_graph is not None and _HAS_NX and "symbol" in df.columns:
        if not isinstance(entity_graph, nx.Graph):
            if isinstance(entity_graph, (str, Path)):
                with open(entity_graph) as f_eg:
                    data = json.load(f_eg)
            else:
                data = entity_graph
            entity_graph = nx.node_link_graph(data)  # type: ignore
        _compute_entity_graph_features(df, feature_names, entity_graph)

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
            df[col] = sym_series.map(lambda s: embeddings.get(s, [0.0] * emb_dim)[i])
        feature_names = feature_names + [f"graph_emb{i}" for i in range(emb_dim)]
    df, feature_names = _apply_symbolic_indicators(df, feature_names, model_json)
    return df, feature_names, embeddings, gnn_state


def _extract_features(
    df: pd.DataFrame | "pl.DataFrame" | "dd.DataFrame",
    feature_names: list[str],
    **kwargs,
) -> tuple[
    pd.DataFrame | "pl.DataFrame" | "dd.DataFrame",
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]:
    """Run enabled feature plugins and return augmented ``df`` and metadata."""
    from .engineering import (  # late import to avoid circular deps
        _CONFIG,
        _FEATURE_RESULTS,
    )

    # Dynamically load any third-party plugins requested via configuration
    load_plugins(_CONFIG.enabled_features)

    if _HAS_DASK and isinstance(df, dd.DataFrame):
        sample = df.head()  # compute small sample for metadata
        sample_out, feature_names, embeddings, gnn_state = _extract_features(
            sample, list(feature_names), **kwargs
        )

        def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
            out, _, _, _ = _extract_features(pdf, list(feature_names), **kwargs)
            return out

        meta = sample_out.iloc[0:0]
        ddf = df.map_partitions(_apply, meta=meta)
        return ddf, feature_names, embeddings, gnn_state

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
