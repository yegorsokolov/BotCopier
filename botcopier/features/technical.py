"""Technical feature extraction utilities."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd

try:  # optional dask support
    import dask.dataframe as dd  # type: ignore

    _HAS_DASK = True
except Exception:  # pragma: no cover - optional
    dd = None  # type: ignore
    _HAS_DASK = False

from .registry import FEATURE_REGISTRY, load_plugins, register_feature

try:  # optional polars dependency
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False
from sklearn.linear_model import LinearRegression

try:  # optional numba dependency
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional

    def njit(*a, **k):  # pragma: no cover - simple stub
        def _decorator(f):
            return f

        return _decorator

    _HAS_NUMBA = False

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
_GNN_STATE: dict[str, list[list[float]]] | None = None

# Lookback windows for fractal metrics
HURST_WINDOW = 50
FRACTAL_DIM_WINDOW = 50

# Parameters for cross-spectral density features
CSD_WINDOW = 16
CSD_FREQ_BINS = 16
_CSD_PARAMS: dict | None = None

_SYMBOLIC_CACHE: dict | None = None


if _HAS_NUMBA:

    @njit(cache=True)
    def _csd_freq_coh_nb(
        sa: np.ndarray, sb: np.ndarray, window: int, freq_bins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.fft.rfftfreq(window)[:freq_bins]
        n = sa.shape[0]
        freq_arr = np.zeros(n)
        coh_arr = np.zeros(n)
        for i in range(window - 1, n):
            xa = sa[i - window + 1 : i + 1]
            xb = sb[i - window + 1 : i + 1]
            fa = np.fft.rfft(xa, freq_bins)
            fb = np.fft.rfft(xb, freq_bins)
            Pxy = fa * np.conjugate(fb)
            Pxx = fa * np.conjugate(fa)
            Pyy = fb * np.conjugate(fb)
            coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-9)
            idx_max = np.argmax(coh)
            freq_arr[i] = freqs[idx_max]
            coh_arr[i] = np.real(coh[idx_max])
        return freq_arr, coh_arr

else:  # pragma: no cover - numba unavailable

    def _csd_freq_coh_nb(
        sa: np.ndarray, sb: np.ndarray, window: int, freq_bins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.fft.rfftfreq(window)[:freq_bins]
        n = sa.shape[0]
        freq_arr = np.zeros(n)
        coh_arr = np.zeros(n)
        for i in range(window - 1, n):
            xa = sa[i - window + 1 : i + 1]
            xb = sb[i - window + 1 : i + 1]
            fa = np.fft.rfft(xa, freq_bins)
            fb = np.fft.rfft(xb, freq_bins)
            Pxy = fa * np.conjugate(fb)
            Pxx = fa * np.conjugate(fa)
            Pyy = fb * np.conjugate(fb)
            coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-9)
            idx_max = np.argmax(coh)
            freq_arr[i] = freqs[idx_max]
            coh_arr[i] = np.real(coh[idx_max])
        return freq_arr, coh_arr


def _csd_pair_impl(
    sa: np.ndarray, sb: np.ndarray, window: int, freq_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    return _csd_freq_coh_nb(sa, sb, window, freq_bins)


FeatureResult = Tuple[
    Any,
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]


_csd_pair = _csd_pair_impl


def _maybe_to_pandas(
    df: pd.DataFrame | "pl.DataFrame",
) -> tuple[pd.DataFrame, str | None]:
    """Return a pandas ``DataFrame`` along with the original backend tag."""

    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.to_pandas(), "polars"
    return df, None


def _restore_frame(df: pd.DataFrame, backend: str | None):
    if backend == "polars" and _HAS_POLARS:
        return pl.from_pandas(df)
    return df


def _load_graph_dataset(
    symbol_graph: dict | str | Path | None,
) -> GraphDataset | None:
    """Return a :class:`GraphDataset` for ``symbol_graph`` when available."""

    if not _HAS_TG or compute_gnn_embeddings is None:
        return None
    if symbol_graph is None or isinstance(symbol_graph, dict):
        return None
    try:
        return GraphDataset(symbol_graph)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load graph dataset from %s", symbol_graph)
    return None


@register_feature("calendar")
def _calendar_feature_plugin(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    calendar_features: bool = True,
    **_: object,
) -> FeatureResult:
    if not calendar_features:
        return df, feature_names, {}, {}

    pdf, backend = _maybe_to_pandas(df)
    if "event_time" not in pdf.columns:
        return df, feature_names, {}, {}

    pdf = pdf.copy()
    pdf["event_time"] = pd.to_datetime(pdf["event_time"], errors="coerce", utc=True)
    if pdf["event_time"].isna().all():
        return df, feature_names, {}, {}

    pdf["hour"] = pdf["event_time"].dt.hour.astype(int)
    pdf["dayofweek"] = pdf["event_time"].dt.dayofweek.astype(int)
    pdf["month"] = pdf["event_time"].dt.month.astype(int)

    pdf["hour_sin"] = np.sin(2 * np.pi * pdf["hour"] / 24.0)
    pdf["hour_cos"] = np.cos(2 * np.pi * pdf["hour"] / 24.0)
    pdf["dow_sin"] = np.sin(2 * np.pi * pdf["dayofweek"] / 7.0)
    pdf["dow_cos"] = np.cos(2 * np.pi * pdf["dayofweek"] / 7.0)
    pdf["month_sin"] = np.sin(2 * np.pi * (pdf["month"] - 1) / 12.0)
    pdf["month_cos"] = np.cos(2 * np.pi * (pdf["month"] - 1) / 12.0)

    for col in [
        "hour",
        "dayofweek",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]:
        if col not in feature_names:
            feature_names.append(col)

    out = _restore_frame(pdf, backend)
    return out, feature_names, {}, {}


@register_feature("lag_diff")
def _lag_diff_feature_plugin(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    lag_windows: Iterable[int] = (1, 5),
    lag_columns: Iterable[str] | None = None,
    **_: object,
) -> FeatureResult:
    pdf, backend = _maybe_to_pandas(df)
    if pdf.empty:
        return df, feature_names, {}, {}

    candidates = (
        [c for c in lag_columns if c in pdf.columns]
        if lag_columns is not None
        else [
            c
            for c in feature_names
            if c in pdf.columns and pd.api.types.is_numeric_dtype(pdf[c])
        ]
    )
    if not candidates:
        return df, feature_names, {}, {}

    symbol_col = pdf["symbol"] if "symbol" in pdf.columns else None
    for col in candidates:
        series = pd.to_numeric(pdf[col], errors="coerce")
        for lag in lag_windows:
            if symbol_col is not None:
                lagged = series.groupby(symbol_col).shift(lag)
            else:
                lagged = series.shift(lag)
            lag_col = f"{col}_lag_{lag}"
            pdf[lag_col] = lagged
            if lag_col not in feature_names:
                feature_names.append(lag_col)

        if symbol_col is not None:
            diff_series = series.groupby(symbol_col).diff()
        else:
            diff_series = series.diff()
        diff_col = f"{col}_diff"
        pdf[diff_col] = diff_series
        if diff_col not in feature_names:
            feature_names.append(diff_col)

    out = _restore_frame(pdf, backend)
    return out, feature_names, {}, {}


def _rolling_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period, min_periods=1).mean()
    roll_down = down.rolling(period, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().fillna(0.0)


@register_feature("technical_indicators")
def _technical_indicator_plugin(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    price_column: str | None = None,
    sma_window: int = 5,
    rsi_period: int = 14,
    macd_short: int = 12,
    macd_long: int = 26,
    macd_signal: int = 9,
    boll_window: int = 20,
    atr_period: int = 14,
    fft_window: int = 16,
    **_: object,
) -> FeatureResult:
    pdf, backend = _maybe_to_pandas(df)
    if pdf.empty:
        return df, feature_names, {}, {}

    price_col = price_column or ("close" if "close" in pdf.columns else "price")
    if price_col not in pdf.columns:
        return df, feature_names, {}, {}

    prices = pd.to_numeric(pdf[price_col], errors="coerce").fillna(0.0)
    symbols = pdf["symbol"] if "symbol" in pdf.columns else None

    ema_short = prices.ewm(span=macd_short, adjust=False).mean()
    ema_long = prices.ewm(span=macd_long, adjust=False).mean()
    macd_val = ema_short - ema_long
    macd_signal_val = macd_val.ewm(span=macd_signal, adjust=False).mean()

    sma = prices.rolling(sma_window, min_periods=1).mean()
    rsi = _rolling_rsi(prices, rsi_period)

    roll_mean = prices.rolling(boll_window, min_periods=1).mean()
    roll_std = prices.rolling(boll_window, min_periods=1).std(ddof=0).fillna(0.0)
    boll_upper = roll_mean + 2.0 * roll_std
    boll_lower = roll_mean - 2.0 * roll_std

    high = pd.to_numeric(pdf.get("high", prices), errors="coerce").fillna(prices)
    low = pd.to_numeric(pdf.get("low", prices), errors="coerce").fillna(prices)
    atr = _rolling_atr(high, low, prices, atr_period)

    sl = pd.to_numeric(pdf.get("sl"), errors="coerce")
    tp = pd.to_numeric(pdf.get("tp"), errors="coerce")
    sl_dist = (prices - sl).abs().fillna(0.0)
    tp_dist = (tp - prices).abs().fillna(0.0)
    sl_dist_atr = np.where(atr > 0, sl_dist / atr, 0.0)
    tp_dist_atr = np.where(atr > 0, tp_dist / atr, 0.0)

    fft_mag0 = pd.Series(0.0, index=pdf.index, dtype=float)
    fft_phase0 = pd.Series(0.0, index=pdf.index, dtype=float)
    fft_mag1 = pd.Series(0.0, index=pdf.index, dtype=float)
    fft_phase1 = pd.Series(0.0, index=pdf.index, dtype=float)

    if symbols is not None:
        group_positions = pdf.groupby(symbols).indices.values()
    else:
        group_positions = [list(range(len(pdf)))]

    for positions in group_positions:
        idx_positions = list(positions)
        if not idx_positions:
            continue
        seq = prices.iloc[idx_positions].to_numpy(dtype=float)
        for pos, iloc_idx in enumerate(idx_positions):
            window = seq[max(0, pos - fft_window + 1) : pos + 1]
            if window.size == 0:
                continue
            fft_vals = np.fft.fft(window, n=4)
            fft_mag0.iloc[iloc_idx] = float(np.abs(fft_vals[0]))
            fft_phase0.iloc[iloc_idx] = float(np.angle(fft_vals[0]))
            fft_mag1.iloc[iloc_idx] = float(np.abs(fft_vals[1]))
            fft_phase1.iloc[iloc_idx] = float(np.angle(fft_vals[1]))

    pdf["sma"] = sma
    pdf["rsi"] = rsi
    pdf["macd"] = macd_val
    pdf["macd_signal"] = macd_signal_val
    pdf["bollinger_upper"] = boll_upper
    pdf["bollinger_middle"] = roll_mean
    pdf["bollinger_lower"] = boll_lower
    pdf["atr"] = atr
    pdf["sl_dist_atr"] = sl_dist_atr
    pdf["tp_dist_atr"] = tp_dist_atr
    pdf["fft_0_mag"] = fft_mag0
    pdf["fft_0_phase"] = fft_phase0
    pdf["fft_1_mag"] = fft_mag1
    pdf["fft_1_phase"] = fft_phase1

    for col in [
        "sma",
        "rsi",
        "macd",
        "macd_signal",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "atr",
        "sl_dist_atr",
        "tp_dist_atr",
        "fft_0_mag",
        "fft_0_phase",
        "fft_1_mag",
        "fft_1_phase",
    ]:
        if col not in feature_names:
            feature_names.append(col)

    out = _restore_frame(pdf, backend)
    return out, feature_names, {}, {}


@register_feature("rolling_correlations")
def _rolling_corr_plugin(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    symbol_graph: dict | str | Path | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    price_column: str = "price",
    **_: object,
) -> FeatureResult:
    if neighbor_corr_windows is None or not neighbor_corr_windows:
        return df, feature_names, {}, {}

    pdf, backend = _maybe_to_pandas(df)
    if "symbol" not in pdf.columns or price_column not in pdf.columns:
        return df, feature_names, {}, {}

    if symbol_graph is None:
        return df, feature_names, {}, {}

    if isinstance(symbol_graph, (str, Path)):
        try:
            graph_data = json.loads(Path(symbol_graph).read_text())
        except Exception:
            return df, feature_names, {}, {}
    else:
        graph_data = symbol_graph

    symbols = graph_data.get("symbols", [])
    adjacency: dict[str, set[str]] = {sym: set() for sym in symbols}

    edge_index = graph_data.get("edge_index")
    if isinstance(edge_index, list) and len(edge_index) == 2:
        for a, b in zip(edge_index[0], edge_index[1]):
            if a < len(symbols) and b < len(symbols):
                sa = symbols[a]
                sb = symbols[b]
                if sa != sb:
                    adjacency.setdefault(sa, set()).add(sb)

    edges = graph_data.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            sa, sb = edge[0], edge[1]
            if sa != sb:
                adjacency.setdefault(sa, set()).add(sb)

    if not adjacency:
        return df, feature_names, {}, {}

    prices = pd.to_numeric(pdf[price_column], errors="coerce").fillna(0.0)
    idx_col = "event_time" if "event_time" in pdf.columns else None
    if idx_col:
        idx_vals = pd.to_datetime(pdf[idx_col], errors="coerce")
    else:
        idx_vals = pd.Series(pdf.index, index=pdf.index)

    temp = pd.DataFrame({
        "_idx": idx_vals,
        "symbol": pdf["symbol"].astype(str),
        "price": prices,
    })

    price_wide = (
        temp.pivot(index="_idx", columns="symbol", values="price")
        .sort_index()
        .ffill()
        .fillna(0.0)
    )
    returns = price_wide.pct_change().fillna(0.0)

    pdf = pdf.copy()
    pdf["_idx"] = temp["_idx"].values

    for base, peers in adjacency.items():
        if base not in returns.columns:
            continue
        for peer in peers:
            if peer not in returns.columns:
                continue
            for win in neighbor_corr_windows:
                if win <= 1:
                    continue
                corr_series = (
                    returns[base]
                    .rolling(int(win), min_periods=2)
                    .corr(returns[peer])
                    .clip(-1.0, 1.0)
                )
                col = f"corr_{base}_{peer}_w{win}"
                mask = pdf["symbol"].astype(str) == base
                aligned = corr_series.reindex(pdf.loc[mask, "_idx"]).fillna(0.0)
                pdf.loc[mask, col] = aligned.to_numpy()
                pdf[col] = pdf[col].fillna(0.0)
                if col not in feature_names:
                    feature_names.append(col)

    pdf.drop(columns=["_idx"], inplace=True)
    out = _restore_frame(pdf, backend)
    return out, feature_names, {}, {}


@register_feature("graph_embeddings")
def _graph_embedding_plugin(
    df: pd.DataFrame | "pl.DataFrame",
    feature_names: list[str],
    *,
    symbol_graph: dict | str | Path | None = None,
    gnn_state: dict | None = None,
    n_jobs: int | None = None,
    **_: object,
) -> FeatureResult:
    """Attach GNN-based symbol embeddings when a graph is provided."""

    gnn_state_out: dict[str, list[list[float]]] = gnn_state or {}
    if symbol_graph is None:
        return df, feature_names, {}, gnn_state_out
    dataset = _load_graph_dataset(symbol_graph)
    if dataset is None:
        return df, feature_names, {}, gnn_state_out

    pdf, backend = _maybe_to_pandas(df)
    pdf = pdf.copy()
    embeddings: dict[str, list[float]] = {}
    executor: ThreadPoolExecutor | None = None

    try:
        if (
            n_jobs is not None
            and n_jobs > 1
        ):
            executor = ThreadPoolExecutor(max_workers=int(n_jobs))
            future = executor.submit(
                compute_gnn_embeddings, pdf.copy(), dataset, state_dict=gnn_state
            )
            embeddings, gnn_state_out = future.result()
        elif compute_gnn_embeddings is not None:
            embeddings, gnn_state_out = compute_gnn_embeddings(
                pdf, dataset, state_dict=gnn_state
            )
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        logger.exception("Failed to compute GNN embeddings")
        embeddings = {}
    finally:
        if executor is not None:
            executor.shutdown()

    if not embeddings or "symbol" not in pdf.columns:
        return _restore_frame(pdf, backend), feature_names, {}, gnn_state_out

    emb_dim = len(next(iter(embeddings.values()), []))
    if emb_dim == 0:
        return _restore_frame(pdf, backend), feature_names, embeddings, gnn_state_out

    sym_series = pdf["symbol"].astype(str)
    for i in range(emb_dim):
        col = f"graph_emb{i}"
        pdf[col] = sym_series.map(lambda s, i=i: embeddings.get(s, [0.0] * emb_dim)[i])
        if col not in feature_names:
            feature_names.append(col)

    out = _restore_frame(pdf, backend)
    global _GNN_STATE
    if gnn_state_out:
        _GNN_STATE = gnn_state_out
    return out, feature_names, embeddings, gnn_state_out


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


def refresh_symbolic_indicators(model_json: Path | str | None = None) -> None:
    """Clear cached symbolic indicators so updates are recognised."""
    global _SYMBOLIC_CACHE
    _SYMBOLIC_CACHE = None
    _load_symbolic_indicators(model_json)


def append_symbolic_indicators(
    formulas: Iterable[str],
    feature_names: Iterable[str],
    model_json: Path | str | None = None,
) -> None:
    """Persist ``formulas`` and ``feature_names`` to ``model_json``.

    This helper ensures that newly discovered indicators are appended to the
    model file while keeping previously stored values intact.  It also refreshes
    the in-memory cache so that subsequent feature extraction picks up the
    updates immediately.
    """

    path = Path(model_json or "model.json")
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # pragma: no cover - corrupt model file
        data = {}

    sym = data.setdefault("symbolic_indicators", {})
    stored_formulas = sym.get("formulas", [])
    stored_feats = sym.get("feature_names", [])

    for f in formulas:
        if f not in stored_formulas:
            stored_formulas.append(f)
    for feat in feature_names:
        if feat not in stored_feats:
            stored_feats.append(feat)

    sym["formulas"] = stored_formulas
    sym["feature_names"] = stored_feats
    path.write_text(json.dumps(data, indent=2))
    refresh_symbolic_indicators(path)


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
    gnn_state: dict | None = None,
    rank_features: bool = False,
    n_jobs: int | None = None,
    model_json: Path | str | None = None,
) -> tuple[
    pd.DataFrame, list[str], dict[str, list[float]], dict[str, list[list[float]]]
]:
    """Attach calendar flags, correlation features and technical signals."""
    from . import engineering as fe  # avoid circular import

    graph_data: dict | None = None
    if symbol_graph is not None:
        if not isinstance(symbol_graph, dict):
            with open(symbol_graph) as f_sg:
                graph_data = json.load(f_sg)
        else:
            graph_data = symbol_graph

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
            freq_arr, coh_arr = _csd_pair(sa, sb, CSD_WINDOW, CSD_FREQ_BINS)
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

    df, feature_names = _apply_symbolic_indicators(df, feature_names, model_json)
    return df, feature_names, {}, {}


def _extract_features(
    df: pd.DataFrame | "pl.DataFrame" | "dd.DataFrame",
    feature_names: list[str],
    *,
    n_jobs: int | None = None,
    **kwargs,
) -> tuple[
    pd.DataFrame | "pl.DataFrame" | "dd.DataFrame",
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]:
    """Run enabled feature plugins and return augmented ``df`` and metadata."""
    from . import engineering as fe  # late import to avoid circular deps
    from .engineering import _CONFIG, _FEATURE_RESULTS

    plugin_overrides_raw = kwargs.pop("plugins", None)
    if plugin_overrides_raw is None:
        plugin_overrides_raw = kwargs.pop("enabled_plugins", None)
    if plugin_overrides_raw is None:
        plugin_overrides_list: list[str] | None = None
    elif isinstance(plugin_overrides_raw, str):
        plugin_overrides_list = [plugin_overrides_raw]
    else:
        plugin_overrides_list = list(plugin_overrides_raw)

    requested = set(_CONFIG.enabled_features)
    if plugin_overrides_list:
        requested.update(plugin_overrides_list)

    # Dynamically load any third-party plugins requested via configuration
    load_plugins(requested)

    if _HAS_DASK and isinstance(df, dd.DataFrame):
        sample = df.head()  # compute small sample for metadata
        sample_kwargs = dict(kwargs)
        if plugin_overrides_list:
            sample_kwargs["plugins"] = list(plugin_overrides_list)
        sample_kwargs.setdefault("n_jobs", n_jobs)
        sample_out, feature_names, embeddings, gnn_state = _extract_features(
            sample, list(feature_names), **sample_kwargs
        )

        def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
            apply_kwargs = dict(kwargs)
            if plugin_overrides_list:
                apply_kwargs["plugins"] = list(plugin_overrides_list)
            apply_kwargs.setdefault("n_jobs", n_jobs)
            out, _, _, _ = _extract_features(
                pdf, list(feature_names), **apply_kwargs
            )
            return out

        meta = sample_out.iloc[0:0]
        ddf = df.map_partitions(_apply, meta=meta)
        return ddf, feature_names, embeddings, gnn_state

    key = id(df)
    if key in _FEATURE_RESULTS:
        logger.info("cache hit for _extract_features")
        return _FEATURE_RESULTS[key]  # type: ignore[return-value]

    base_plugins = [
        "calendar",
        "lag_diff",
        "technical_indicators",
        "rolling_correlations",
        "technical",
        "graph_embeddings",
    ]

    plugins: list[str] = []
    for name in base_plugins:
        if name in FEATURE_REGISTRY and name not in plugins:
            plugins.append(name)

    for name in requested:
        if name in FEATURE_REGISTRY and name not in plugins:
            plugins.append(name)

    if "technical" not in plugins and "technical" in FEATURE_REGISTRY:
        plugins.append("technical")

    sequential_plugins = [name for name in plugins if name in base_plugins]
    optional_plugins = [name for name in plugins if name not in sequential_plugins]

    embeddings: dict[str, list[float]] = {}
    gnn_state: dict[str, list[list[float]]] = {}
    calendar_executed = False
    for name in sequential_plugins:
        func = FEATURE_REGISTRY.get(name)
        if func is None:
            logger.warning("Feature plugin %s not found", name)
            continue
        plugin_kwargs = dict(kwargs)
        if name == "technical" and calendar_executed:
            plugin_kwargs["calendar_features"] = False
        df, feature_names, emb, gnn = func(df, feature_names, **plugin_kwargs)
        embeddings.update(emb or {})
        gnn_state.update(gnn or {})
        if name == "calendar":
            calendar_executed = True

    if optional_plugins:
        df, feature_names, emb_extra, gnn_extra = fe._apply_parallel_plugins(
            df,
            feature_names,
            optional_plugins,
            kwargs=kwargs,
            n_jobs=n_jobs,
            calendar_executed=calendar_executed,
        )
        embeddings.update(emb_extra)
        gnn_state.update(gnn_extra)
    if embeddings:
        emb_dim = len(next(iter(embeddings.values())))
        for i in range(emb_dim):
            col = f"graph_emb{i}"
            if col not in feature_names:
                feature_names.append(col)
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
