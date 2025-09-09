#!/usr/bin/env python3
"""Train a lightweight logistic regression model from Python trade logs.

The original project consumed CSV exports from an MQL4 Expert Advisor.  The
new observer writes ``logs/trades_raw.csv`` directly from Python with features
already available as numeric columns.  This module reads those logs, normalises
the feature columns and fits a simple model whose parameters are stored in
``model.json``.

Only a very small subset of the original functionality is retained which keeps
resource detection utilities and the federated ``sync_with_server`` helper.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
import shutil
import importlib.util
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple, Callable, Sequence

import logging
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import set_config
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from .splitters import PurgedWalkForward
from sklearn.calibration import CalibratedClassifierCV
from .model_fitting import (
    fit_xgb_classifier,
    fit_lgbm_classifier,
    fit_catboost_classifier,
    _compute_decay_weights,
)
from .meta_strategy import train_meta_model
from .meta_adapt import _logistic_grad, _sigmoid

try:  # Optional dependency
    import optuna

    _HAS_OPTUNA = True
except Exception:  # pragma: no cover
    optuna = None  # type: ignore
    _HAS_OPTUNA = False

try:  # Optional dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

try:  # Optional dependency
    import shap

    _HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None  # type: ignore
    _HAS_SHAP = False

try:  # Optional dependency
    from cleanlab.rank import get_label_quality_scores

    _HAS_CLEANLAB = True
except Exception:  # pragma: no cover
    get_label_quality_scores = None  # type: ignore
    _HAS_CLEANLAB = False

try:  # Optional dependency
    from imblearn.over_sampling import SMOTE

    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    SMOTE = None  # type: ignore
    _HAS_IMBLEARN = False

from .features import _sma, _rsi, _bollinger, _macd_update, _atr

try:  # Optional graph dependencies
    from .graph_dataset import GraphDataset, compute_gnn_embeddings

    _HAS_TG = True
except Exception:  # pragma: no cover
    GraphDataset = None  # type: ignore
    compute_gnn_embeddings = None  # type: ignore
    _HAS_TG = False


# Quantile bounds for feature clipping
_CLIP_BOUNDS = (0.01, 0.99)


def _clip_train_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip extreme feature values and return clipped array with bounds."""
    low = np.quantile(X, _CLIP_BOUNDS[0], axis=0)
    high = np.quantile(X, _CLIP_BOUNDS[1], axis=0)
    return np.clip(X, low, high), low, high


def _clip_apply(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Apply precomputed clipping bounds to features."""
    return np.clip(X, low, high)


if _HAS_TORCH:

    class _TTBlock(torch.nn.Module):
        """Attention block used by :class:`TabTransformer`."""

        def __init__(
            self, dim: int, heads: int, ff_dim: int, dropout: float
        ) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(dim, ff_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ff_dim, dim),
            )
            self.norm1 = torch.nn.LayerNorm(dim)
            self.norm2 = torch.nn.LayerNorm(dim)
            self.drop1 = torch.nn.Dropout(dropout)
            self.drop2 = torch.nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised via higher level tests
            attn, _ = self.attn(x, x, x)
            x = self.norm1(x + self.drop1(attn))
            ff = self.ff(x)
            return self.norm2(x + self.drop2(ff))


    class TabTransformer(torch.nn.Module):
        """Simple tabular transformer with multi-head attention layers."""

        def __init__(
            self,
            num_features: int,
            dim: int = 32,
            heads: int = 4,
            depth: int = 2,
            ff_dim: int = 64,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.embed = torch.nn.Linear(1, dim)
            self.layers = torch.nn.ModuleList(
                [_TTBlock(dim, heads, ff_dim, dropout) for _ in range(depth)]
            )
            self.norm = torch.nn.LayerNorm(dim)
            self.head = torch.nn.Linear(num_features * dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised via helper
            x = self.embed(x.unsqueeze(-1))
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            x = x.reshape(x.size(0), -1)
            return self.head(x)


    def fit_tab_transformer(
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        dropout: float = 0.0,
        device: str = "cpu",
    ) -> tuple[dict[str, list], Callable[[np.ndarray], np.ndarray], TabTransformer]:
        """Train a :class:`TabTransformer` on ``X`` and ``y``."""
        dev = torch.device(device)
        model = TabTransformer(X.shape[1], dropout=dropout).to(dev)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(max(1, epochs)):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()

        def _predict(inp: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                xt = torch.tensor(inp, dtype=torch.float32, device=dev)
                logits = model(xt)
                return torch.sigmoid(logits).cpu().numpy().ravel()

        state = {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}
        return state, _predict, model

else:  # pragma: no cover - torch is optional

    class TabTransformer:  # type: ignore
        pass

    def fit_tab_transformer(*_args, **_kwargs):  # type: ignore
        raise ImportError("PyTorch is required for TabTransformer")


def _apply_drift_pruning(
    df: pd.DataFrame,
    feature_names: list[str],
    drift_scores: dict[str, float] | None,
    threshold: float,
    weight: float,
) -> list[str]:
    """Drop or down-weight features whose drift score exceeds ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing feature columns.
    feature_names : list[str]
        Current list of feature names.
    drift_scores : dict[str, float] | None
        Mapping of feature name to drift score.
    threshold : float
        Features with scores above this value are pruned.
    weight : float
        If ``weight`` > 0, features are multiplied by this factor instead of
        being dropped.

    Returns
    -------
    list[str]
        The list of features that were removed or down-weighted.
    """
    removed: list[str] = []
    if not drift_scores or threshold <= 0:
        return removed
    for feat in list(feature_names):
        score = drift_scores.get(feat)
        if score is None or score <= threshold:
            continue
        if weight <= 0:
            df.drop(columns=feat, inplace=True)
            feature_names.remove(feat)
        else:
            df[feat] = df[feat] * weight
        removed.append(feat)
    if removed:
        logging.info("Drift pruning applied to features: %s", removed)
    return removed


def _maybe_smote(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling when class imbalance exceeds ``threshold``."""
    if not _HAS_IMBLEARN:
        return X, y, w
    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        return X, y, w
    minor = class_counts.min()
    major = class_counts.max()
    if minor <= 1 or major / max(minor, 1) <= threshold:
        return X, y, w
    k_neighbors = min(5, minor - 1)
    sm = SMOTE(k_neighbors=k_neighbors, random_state=0)
    X_res, y_res = sm.fit_resample(X, y)
    minority_class = int(np.argmin(class_counts))
    minority_weight = float(w[y == minority_class].mean()) if len(w) else 1.0
    extra_w = np.full(len(X_res) - len(X), minority_weight, dtype=float)
    w_res = np.concatenate([w, extra_w])
    return X_res, y_res, w_res


def _train_autoencoder_weights(
    X: np.ndarray,
    out_file: Path,
    *,
    latent_dim: int = 8,
    epochs: int = 10,
    device: str = "cpu",
) -> None:
    """Train a linear autoencoder via SVD and save the encoder weights.

    A simple approximation of an autoencoder is obtained by performing singular
    value decomposition on the feature matrix and keeping the top ``latent_dim``
    right singular vectors.  These act as the encoder weights and are stored as
    a NumPy ``.npy`` array with ``.pt`` extension for compatibility.
    """
    _ = epochs, device  # unused but kept for API compatibility
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    weights = Vt[:latent_dim].astype(np.float32)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        np.save(f, weights)


def _encode_with_autoencoder(
    X: np.ndarray, weight_file: Path, *, device: str = "cpu"
) -> np.ndarray:
    """Load encoder weights saved by :func:`_train_autoencoder_weights`.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix to transform.
    weight_file : Path
        Path to ``autoencoder.pt`` produced during training.
    device : str
        Unused but kept for API compatibility with potential torch version.
    """
    _ = device  # unused
    weights = np.load(weight_file)
    return X.dot(weights.T)


def _persist_encoder_meta(out_dir: Path, model: dict, enc_path: Path | None) -> None:
    """Copy contrastive encoder files and record metadata in ``model``.

    Parameters
    ----------
    out_dir : Path
        Directory where ``model.json`` will be written.
    model : dict
        Model dictionary to update with encoder metadata.
    enc_path : Path | None
        Source ``encoder.pt`` produced by :mod:`pretrain_contrastive`.
    """

    if not enc_path:
        return
    try:
        enc_src = Path(enc_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_dst = out_dir / "encoder.pt"
        shutil.copy(enc_src, pt_dst)
        onnx_src = enc_src.with_suffix(".onnx")
        meta = {"file": "encoder.pt"}
        if onnx_src.exists():
            shutil.copy(onnx_src, out_dir / "encoder.onnx")
            meta["onnx"] = "encoder.onnx"
        if _HAS_TORCH:
            state = torch.load(enc_src, map_location="cpu")
            meta.update(
                {
                    "window": int(state.get("window", 0)),
                    "dim": int(state.get("dim", 0)),
                }
            )
        model["encoder"] = meta
    except Exception:  # pragma: no cover - best effort
        pass


def _mean_abs_shap_linear(
    clf: SGDClassifier, scaler: RobustScaler, X: np.ndarray
) -> np.ndarray:
    """Return mean absolute SHAP values for linear models."""
    if _HAS_SHAP:
        explainer = shap.LinearExplainer(clf, scaler.transform(X))
        shap_vals = explainer.shap_values(scaler.transform(X))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
    else:
        coeffs = clf.coef_[0]
        shap_vals = ((X - scaler.center_) / scaler.scale_) * coeffs
    return np.mean(np.abs(shap_vals), axis=0)


def _mean_abs_shap_tree(clf, X: np.ndarray) -> np.ndarray:
    """Return mean absolute SHAP values for tree models."""
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            return np.mean(np.abs(shap_vals), axis=0)
        except Exception:  # pragma: no cover
            logging.debug("Tree SHAP computation failed", exc_info=True)
    return np.zeros(X.shape[1])


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


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
) -> Tuple[Iterable[pd.DataFrame] | pd.DataFrame, list[str], list[str]]:
    """Load trade logs from ``trades_raw.csv``.

    Parameters are kept for backwards compatibility with the previous API but
    only local CSV files are supported now.
    """
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

    # ------------------------------------------------------------------
    # Create additional forward looking label columns based on future PnL
    # ------------------------------------------------------------------
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
        horizon_idx = (
            np.arange(len(df)) + int(hold_period)
        ).clip(0, len(df) - 1)
        meta = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            end = int(horizon_idx[i])
            meta_i = 0.0
            for j in range(i + 1, end + 1):
                p = prices.iloc[j]
                if p >= tp.iloc[i]:
                    meta_i = 1.0
                    break
                if p <= sl.iloc[i]:
                    meta_i = 0.0
                    break
            meta[i] = meta_i
        df["take_profit"] = tp
        df["stop_loss"] = sl
        df["horizon"] = horizon_idx
        df["meta_label"] = meta
        for horizon in (5, 20):
            label_name = f"label_h{horizon}"
            if label_name not in df.columns:
                pnl = prices.shift(-horizon) - prices
                df[label_name] = (pnl > 0).astype(float).fillna(0.0)

    # When ``chunk_size`` is provided (or lite_mode explicitly enabled), yield
    # DataFrame chunks instead of a single concatenated frame so callers can
    # control memory usage.  ``lite_mode`` keeps backwards compatibility where
    # chunked iteration was previously the default behaviour.
    cs = chunk_size or (50000 if lite_mode else None)
    if cs:

        def _iter():
            for start in range(0, len(df), cs):
                yield df.iloc[start : start + cs]

        return _iter(), feature_cols, []

    return df, feature_cols, []


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _extract_features(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    event_window: float = 60.0,
    news_sentiment: pd.DataFrame | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    regime_model: dict | str | Path | None = None,
    tick_encoder: Path | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[float]], dict[str, list[list[float]]]]:
    """Attach graph embeddings, calendar flags and correlation features."""

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
                    if len(use_cols) == window:
                        X = torch.tensor(
                            df[use_cols].to_numpy(dtype=float), dtype=torch.float32
                        )
                        emb = X @ weight_t
                        for i in range(weight_t.shape[1]):
                            col = f"enc_{i}"
                            df[col] = emb[:, i].numpy()
                        feature_names = feature_names + [
                            f"enc_{i}" for i in range(weight_t.shape[1])
                        ]
            except Exception:
                pass

    if calendar_file is not None and "event_time" in df.columns:
        try:
            cdf = pd.read_csv(calendar_file)
            cdf.columns = [c.lower() for c in cdf.columns]
            cdf["time"] = pd.to_datetime(cdf["time"], errors="coerce")
            events = list(
                zip(cdf["time"], pd.to_numeric(cdf.get("impact", 0), errors="coerce"))
            )
        except Exception:
            events = []
        df["event_flag"] = 0.0
        df["event_impact"] = 0.0
        if events:
            for ev_time, ev_imp in events:
                mask = (
                    df["event_time"].sub(ev_time).abs().dt.total_seconds()
                    <= event_window * 60.0
                )
                df.loc[mask, "event_flag"] = 1.0
                df.loc[mask, "event_impact"] = np.maximum(
                    df.loc[mask, "event_impact"], ev_imp
                )
        feature_names = feature_names + ["event_flag", "event_impact"]

    if (
        news_sentiment is not None
        and "event_time" in df.columns
        and "symbol" in df.columns
        and len(news_sentiment) > 0
    ):
        ns = news_sentiment.copy()
        ns.columns = [c.lower() for c in ns.columns]
        ns["timestamp"] = pd.to_datetime(ns["timestamp"], errors="coerce")
        ns.sort_values(["symbol", "timestamp"], inplace=True)
        df_idx = df.index
        df = df.sort_values(["symbol", "event_time"])
        merged = pd.merge_asof(
            df,
            ns,
            left_on="event_time",
            right_on="timestamp",
            by="symbol",
            direction="nearest",
        )
        merged["sentiment_score"] = merged["score"].fillna(0.0)
        merged.drop(columns=["timestamp", "score"], inplace=True)
        df = merged.set_index(df_idx).sort_index()
        feature_names = feature_names + ["sentiment_score"]
    price_col = next((c for c in ["price", "bid", "ask"] if c in df.columns), None)
    if price_col is not None:
        price_series = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
        # Compute rolling standard deviation as a simple volatility estimate
        df["price_volatility"] = (
            price_series.rolling(window=30, min_periods=1).std().fillna(0.0)
        )

        prices: list[float] = []
        macd_state: dict[str, float] = {}
        sma_vals: list[float] = []
        rsi_vals: list[float] = []
        macd_vals: list[float] = []
        macd_sig_vals: list[float] = []
        boll_u: list[float] = []
        boll_m: list[float] = []
        boll_l: list[float] = []
        atr_vals: list[float] = []
        for val in price_series:
            prices.append(float(val))
            sma_vals.append(_sma(prices, 5))
            rsi_vals.append(_rsi(prices, 14))
            macd, sig = _macd_update(macd_state, prices[-1])
            macd_vals.append(macd)
            macd_sig_vals.append(sig)
            u, m, l = _bollinger(prices, 20)
            boll_u.append(u)
            boll_m.append(m)
            boll_l.append(l)
            atr_vals.append(_atr(prices, 14))
        df["sma"] = sma_vals
        df["rsi"] = rsi_vals
        df["macd"] = macd_vals
        df["macd_signal"] = macd_sig_vals
        df["bollinger_upper"] = boll_u
        df["bollinger_middle"] = boll_m
        df["bollinger_lower"] = boll_l
        df["atr"] = atr_vals
        feature_names = feature_names + [
            "sma",
            "rsi",
            "macd",
            "macd_signal",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
            "atr",
        ]

        # Fourier features from rolling windows of prices
        fft_window = 16
        fft_bins = 3
        price_win: deque[float] = deque(maxlen=fft_window)
        fft_mag = [[0.0] * len(price_series) for _ in range(fft_bins)]
        fft_phase = [[0.0] * len(price_series) for _ in range(fft_bins)]
        for idx, val in enumerate(price_series):
            price_win.append(float(val))
            if len(price_win) == fft_window:
                fft_vals = np.fft.rfft(np.array(price_win, dtype=float))
                for i in range(fft_bins):
                    j = i + 1  # skip the DC component
                    if j < len(fft_vals):
                        comp = fft_vals[j]
                        fft_mag[i][idx] = float(np.abs(comp))
                        fft_phase[i][idx] = float(np.angle(comp))
        for i in range(fft_bins):
            df[f"fft_{i}_mag"] = fft_mag[i]
            df[f"fft_{i}_phase"] = fft_phase[i]
        feature_names = feature_names + [
            f"fft_{i}_mag" for i in range(fft_bins)
        ] + [f"fft_{i}_phase" for i in range(fft_bins)]
    base_cols: dict[str, str] = {}
    if price_col is not None:
        base_cols["price"] = price_col
    for col in ["volume", "spread"]:
        if col in df.columns:
            base_cols[col] = col
    for name, src in base_cols.items():
        series = pd.to_numeric(df[src], errors="coerce")
        df[f"{name}_lag_1"] = series.shift(1).fillna(0.0)
        df[f"{name}_lag_5"] = series.shift(5).fillna(0.0)
        df[f"{name}_diff"] = series.diff().fillna(0.0)
        feature_names.extend(
            [f"{name}_lag_1", f"{name}_lag_5", f"{name}_diff"]
        )

    # Rolling correlations with graph neighbors
    if (
        neighbor_corr_windows
        and graph_data is not None
        and price_col is not None
        and "symbol" in df.columns
    ):
        windows = list(neighbor_corr_windows)
        symbols = graph_data.get("symbols") or list(
            graph_data.get("nodes", {}).keys()
        )
        edge_index = graph_data.get("edge_index") or []
        adjacency: dict[str, set[str]] = {sym: set() for sym in symbols}
        if isinstance(edge_index, list) and len(edge_index) == 2:
            for s_idx, d_idx in zip(edge_index[0], edge_index[1]):
                try:
                    src = symbols[int(s_idx)]
                    dst = symbols[int(d_idx)]
                    adjacency.setdefault(src, set()).add(dst)
                except Exception:  # pragma: no cover - malformed graph
                    continue
        if not edge_index and graph_data.get("cointegration"):
            for src, nbrs in graph_data["cointegration"].items():
                adjacency.setdefault(src, set()).update(nbrs.keys())

        idx_col = "event_time" if "event_time" in df.columns else None
        idx_series = df[idx_col] if idx_col else df.index
        wide = df.pivot_table(index=idx_series, columns="symbol", values=price_col)
        corr_frames: list[pd.Series] = []
        base_map: dict[str, str] = {}
        for src, nbrs in adjacency.items():
            if src not in wide.columns:
                continue
            for dst in nbrs:
                if dst not in wide.columns:
                    continue
                for win in windows:
                    corr = wide[src].rolling(window=win, min_periods=2).corr(wide[dst])
                    col = f"corr_{src}_{dst}_w{win}"
                    corr.name = col
                    corr_frames.append(corr)
                    base_map[col] = src
        if corr_frames:
            corr_df = pd.concat(corr_frames, axis=1)
            if idx_col:
                df = df.merge(corr_df, left_on=idx_col, right_index=True, how="left")
            else:
                df = df.merge(corr_df, left_index=True, right_index=True, how="left")
            for col, src in base_map.items():
                df[col] = np.where(df["symbol"] == src, df[col], 0.0)
                df[col] = df[col].fillna(0.0)
            feature_names.extend(base_map.keys())

    if regime_model is not None:
        try:
            if not isinstance(regime_model, dict):
                with open(regime_model) as f_rm:
                    rm = json.load(f_rm)
            else:
                rm = regime_model
            r_feats = rm.get("feature_names", [])
            centers = np.asarray(rm.get("centers", []), dtype=float)
            mean = np.asarray(rm.get("mean", []), dtype=float)
            std = np.asarray(rm.get("std", []), dtype=float)
            std[std == 0] = 1.0
            if (
                len(r_feats)
                and centers.size
                and all(col in df.columns for col in r_feats)
            ):
                X = df[r_feats].astype(float).to_numpy()
                X_scaled = (X - mean) / std
                dists = np.linalg.norm(
                    X_scaled[:, None, :] - centers[None, :, :], axis=2
                )
                labels = np.argmin(dists, axis=1).astype(int)
                df["regime"] = labels
                one_hot_cols = []
                for i in range(centers.shape[0]):
                    col = f"regime_{i}"
                    df[col] = (labels == i).astype(float)
                    one_hot_cols.append(col)
                for col in one_hot_cols:
                    if col not in feature_names:
                        feature_names.append(col)
        except Exception:
            pass
    # Generate pairwise interaction features from numeric columns
    # Disabled by default to keep the feature space stable across environments
    # where ``scikit-learn`` may or may not be installed.

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
                df[col] = sym_series.map(
                    lambda s: embeddings.get(s, [0.0] * emb_dim)[i]
                )
            feature_names = feature_names + [f"graph_emb{i}" for i in range(emb_dim)]

    return df, feature_names, embeddings, gnn_state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_multi_output_clf(
    df: pd.DataFrame,
    feature_names: list[str],
    label_cols: list[str],
    out_dir: Path,
) -> None:
    """Fit a multi-output ``SGDClassifier`` and persist model parameters."""

    X_all = df[feature_names].to_numpy(dtype=float)
    y_all = df[label_cols].astype(int).to_numpy()
    w_all = df["sample_weight"].to_numpy(dtype=float)

    X_all, c_low, c_high = _clip_train_features(X_all)
    scaler = RobustScaler().fit(X_all)
    base = SGDClassifier(loss="log_loss")
    try:  # sklearn >=1.4 metadata routing
        base.set_fit_request(sample_weight=True)
    except Exception:  # pragma: no cover - older sklearn
        pass
    clf = MultiOutputClassifier(base)
    clf.fit(scaler.transform(X_all), y_all, sample_weight=w_all)

    coefs = np.vstack([est.coef_[0] for est in clf.estimators_])
    intercepts = np.array([est.intercept_[0] for est in clf.estimators_])

    params = {
        "coefficients": coefs.astype(float).tolist(),
        "intercept": intercepts.astype(float).tolist(),
        "threshold": 0.5,
        "thresholds": {name: 0.5 for name in label_cols},
        "feature_mean": scaler.center_.astype(float).tolist(),
        "feature_std": scaler.scale_.astype(float).tolist(),
        "clip_low": c_low.astype(float).tolist(),
        "clip_high": c_high.astype(float).tolist(),
    }
    model = {
        "models": {"sgd": params},
        "feature_names": feature_names,
        "label_columns": label_cols,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)


def _train_lite_mode(
    data_dir: Path,
    out_dir: Path,
    *,
    chunk_size: int | None = None,
    hash_size: int = 0,
    flight_uri: str | None = None,
    mode: str = "lite",
    min_accuracy: float = 0.0,
    min_profit: float = 0.0,
    extra_prices: dict[str, Iterable[float]] | None = None,
    replay_file: Path | None = None,
    replay_weight: float = 1.0,
    uncertain_file: Path | None = None,
    uncertain_weight: float = 2.0,
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    optuna_trials: int = 0,
    optuna_folds: int = 3,
    half_life_days: float = 0.0,
    prune_threshold: float = 0.0,
    logreg_c_low: float = 0.01,
    logreg_c_high: float = 100.0,
    logreg_l1_low: float = 0.0,
    logreg_l1_high: float = 1.0,
    gboost_n_estimators_low: int = 50,
    gboost_n_estimators_high: int = 200,
    gboost_subsample_low: float = 0.5,
    gboost_subsample_high: float = 1.0,
    purge_gap: int = 1,
    news_sentiment: pd.DataFrame | None = None,
    ensemble: str | None = None,
    mi_threshold: float = 0.0,
    neighbor_corr_windows: Iterable[int] | None = None,
    use_volatility_weight: bool = False,
    use_profit_weight: bool = False,
    regime_model: dict | str | Path | None = None,
    per_regime: bool = False,
    filter_noise: bool = False,
    noise_quantile: float = 0.9,
    smote_threshold: float | None = None,
    use_autoencoder: bool = False,
    autoencoder_dim: int = 8,
    autoencoder_epochs: int = 10,
    device: str = "cpu",
    synthetic_data: Path | pd.DataFrame | None = None,
    synthetic_weight: float = 0.2,
    meta_weights: Sequence[float] | Path | None = None,
    tick_encoder: Path | None = None,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 0.0,
    drift_weight: float = 0.0,
    bayesian_ensembles: int = 0,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    use_meta_label: bool = False,
    **_: object,
) -> None:
    """Train ``SGDClassifier`` on features from ``trades_raw.csv``."""
    df, feature_names, _ = _load_logs(
        data_dir,
        chunk_size=chunk_size,
        flight_uri=flight_uri,
        take_profit_mult=take_profit_mult,
        stop_loss_mult=stop_loss_mult,
        hold_period=hold_period,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    feature_names = list(feature_names)

    if use_meta_label and "meta_label" in df.columns:
        df["label"] = df["meta_label"]

    meta_init: np.ndarray | None = None
    if meta_weights is not None:
        if isinstance(meta_weights, (str, Path)):
            try:
                meta_data = json.loads(Path(meta_weights).read_text())
                meta_init = np.array(meta_data.get("meta_weights", meta_data), dtype=float)
            except Exception:
                meta_init = np.array(meta_weights, dtype=float)
        else:
            meta_init = np.array(list(meta_weights), dtype=float)

    df["synthetic"] = 0.0
    if synthetic_data is not None:
        try:
            sdf = (
                synthetic_data.copy()
                if isinstance(synthetic_data, pd.DataFrame)
                else pd.read_csv(synthetic_data)
            )
            sdf.columns = [c.lower() for c in sdf.columns]
            if "label" not in sdf.columns:
                price_col = next((c for c in ["price", "bid", "ask"] if c in sdf.columns), None)
                if price_col is not None:
                    prices = pd.to_numeric(sdf[price_col], errors="coerce").fillna(0.0)
                    sdf["label"] = (prices.diff().fillna(0.0) > 0).astype(float)
                else:
                    sdf["label"] = 0.0
            sdf["synthetic"] = 1.0
            for col in df.columns:
                if col not in sdf.columns:
                    sdf[col] = 0.0
            sdf = sdf[df.columns]
            df = pd.concat([df, sdf], ignore_index=True)
        except Exception:
            pass

    if replay_file:
        rdf = pd.read_csv(replay_file)
        rdf.columns = [c.lower() for c in rdf.columns]
        df = pd.concat([df, rdf], ignore_index=True)
    unc_df = None
    uncertain_count = 0
    uncertain_session = 0
    if uncertain_file:
        try:
            unc_df = pd.read_csv(uncertain_file, sep=";")
            unc_df.columns = [c.lower() for c in unc_df.columns]
            if "features" in unc_df.columns:
                feats = unc_df["features"].str.split(":")
                feat_df = pd.DataFrame(feats.tolist(), columns=feature_names)
                unc_df = pd.concat([unc_df, feat_df], axis=1)
            for col in df.columns:
                if col not in unc_df.columns:
                    unc_df[col] = 0.0
            unc_df = unc_df[df.columns]
            uncertain_count = len(unc_df)
            if "session" in unc_df.columns and len(unc_df):
                uncertain_session = int(pd.to_numeric(unc_df["session"], errors="coerce").fillna(0).iloc[0])
            df = pd.concat([df, unc_df], ignore_index=True)
        except Exception:
            unc_df = None
    if use_profit_weight and ("profit" in df.columns or "net_profit" in df.columns):
        profit_col = "profit" if "profit" in df.columns else "net_profit"
        weights = pd.to_numeric(df[profit_col], errors="coerce").abs().to_numpy()
    elif "lots" in df.columns:
        weights = pd.to_numeric(df["lots"], errors="coerce").abs().to_numpy()
    else:
        weights = np.ones(len(df), dtype=float)
    if "lots" in df.columns:
        lot_vals = pd.to_numeric(df["lots"], errors="coerce").abs().to_numpy()
        weights = np.where(weights > 0, weights, lot_vals)
    if replay_file:
        weights[-len(rdf) :] *= replay_weight
    if uncertain_file and uncertain_count:
        weights[-uncertain_count:] *= uncertain_weight

    # Compute sample age and optional exponential decay weights
    if "event_time" in df.columns:
        event_times = df["event_time"].to_numpy(dtype="datetime64[s]")
        if half_life_days > 0:
            age_days, decay = _compute_decay_weights(event_times, half_life_days)
            weights = weights * decay
        else:
            ref_time = event_times.max()
            age_days = (
                (ref_time - event_times).astype("timedelta64[s]").astype(float)
                / (24 * 3600)
            )
    else:
        age_days = (df.index.max() - df.index).astype(float)
        if half_life_days > 0:
            decay = 0.5 ** (age_days / half_life_days)
            weights = weights * decay
    df["age_days"] = age_days

    # Normalise weights to mean 1 for stable training
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w

    df["sample_weight"] = weights
    if "synthetic" in df.columns:
        df.loc[df["synthetic"] > 0, "sample_weight"] *= synthetic_weight

    if "label" not in df.columns:
        raise ValueError("label column missing from data")

    df, feature_names, embeddings, gnn_state = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
        neighbor_corr_windows=neighbor_corr_windows,
        regime_model=regime_model,
        tick_encoder=tick_encoder,
    )
    if use_volatility_weight and "price_volatility" in df.columns:
        vol = pd.to_numeric(df["price_volatility"], errors="coerce").fillna(0.0)
        mean_vol = float(vol.mean())
        if mean_vol > 0:
            df["sample_weight"] = df["sample_weight"] * (vol / mean_vol)
    # Re-normalise weights after all adjustments
    sw = df["sample_weight"].to_numpy(dtype=float)
    mean_sw = float(np.mean(sw))
    if mean_sw > 0:
        df["sample_weight"] = sw / mean_sw
    feature_names = [
        c
        for c in feature_names
        if c not in {"label", "profit", "net_profit", "hour", "day_of_week", "symbol"}
    ]

    if extra_prices:
        price_col = next((c for c in ["price", "bid", "ask"] if c in df.columns), None)
        if price_col:
            base_series = pd.to_numeric(df[price_col], errors="coerce")
            base_symbol = (
                str(df.get("symbol", pd.Series(["base"])).iloc[0])
                if len(df.get("symbol", pd.Series(["base"])).unique()) == 1
                else "base"
            )
            for sym, series in extra_prices.items():
                peer = pd.Series(list(series), index=df.index, dtype=float)
                corr = base_series.rolling(window=5, min_periods=1).corr(peer)
                ratio = base_series / peer.replace(0, np.nan)
                corr_name = f"corr_{base_symbol}_{sym}"
                ratio_name = f"ratio_{base_symbol}_{sym}"
                df[corr_name] = corr.fillna(0.0)
                df[ratio_name] = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        feature_names = [
            c
            for c in df.columns
            if c
            not in {"label", "profit", "net_profit", "hour", "day_of_week", "symbol"}
        ]
    # Compute mutual information for each feature and drop low-score features
    X_mi = df[feature_names].to_numpy(dtype=float)
    y_mi = df["label"].astype(int).to_numpy()
    X_mi, _, _ = _clip_train_features(X_mi)
    scaler_mi = RobustScaler().fit(X_mi)
    mi_scores = mutual_info_classif(scaler_mi.transform(X_mi), y_mi, random_state=0)
    retained_features = [
        name for name, score in zip(feature_names, mi_scores) if score >= mi_threshold
    ]
    if len(retained_features) < len(feature_names):
        dropped = [f for f in feature_names if f not in retained_features]
        logging.info(
            "Dropping %d features below MI threshold %.3f: %s",
            len(dropped),
            mi_threshold,
            dropped,
        )
    feature_names = retained_features
    logging.info("Retained features after MI filtering: %s", feature_names)

    _apply_drift_pruning(
        df, feature_names, drift_scores, drift_threshold, drift_weight
    )

    dropped_noisy = 0
    if filter_noise:
        X_nf = df[feature_names].to_numpy()
        y_nf = df["label"].astype(int).to_numpy()
        sw_nf = df["sample_weight"].to_numpy()
        n_splits_nf = min(5, len(df) - 1)
        if n_splits_nf >= 2:
            tscv_nf = PurgedWalkForward(n_splits=n_splits_nf, gap=purge_gap)
            probs = np.zeros(len(df))
            for tr_idx, va_idx in tscv_nf.split(X_nf):
                X_tr, X_va = X_nf[tr_idx], X_nf[va_idx]
                y_tr = y_nf[tr_idx]
                w_tr = sw_nf[tr_idx]
                X_tr, low, high = _clip_train_features(X_tr)
                X_va = _clip_apply(X_va, low, high)
                scaler_nf = RobustScaler().fit(X_tr)
                clf_nf = SGDClassifier(loss="log_loss")
                clf_nf.partial_fit(
                    scaler_nf.transform(X_tr),
                    y_tr,
                    classes=np.array([0, 1]),
                    sample_weight=w_tr,
                )
                probs[va_idx] = clf_nf.predict_proba(
                    scaler_nf.transform(X_va)
                )[:, 1]
        else:
            X_nf_clip, low, high = _clip_train_features(X_nf)
            scaler_nf = RobustScaler().fit(X_nf_clip)
            clf_nf = SGDClassifier(loss="log_loss")
            clf_nf.partial_fit(
                scaler_nf.transform(X_nf_clip),
                y_nf,
                classes=np.array([0, 1]),
                sample_weight=sw_nf,
            )
            probs = clf_nf.predict_proba(
                scaler_nf.transform(X_nf_clip)
            )[:, 1]
        errors = np.abs(probs - y_nf)
        drop_count = int(max(1, (1 - noise_quantile) * len(errors)))
        if _HAS_CLEANLAB:
            try:
                psx = np.vstack([1 - probs, probs]).T
                quality = get_label_quality_scores(
                    y_nf, psx, sample_weight=sw_nf
                )
                drop_idx = np.argpartition(quality, drop_count)[:drop_count]
                keep_mask = np.ones(len(quality), dtype=bool)
                keep_mask[drop_idx] = False
                dropped_noisy = int(len(keep_mask) - keep_mask.sum())
                df = df.loc[keep_mask].reset_index(drop=True)
                logging.info(
                    "Dropped %d noisy samples using cleanlab", dropped_noisy
                )
            except Exception:
                drop_idx = np.argpartition(errors, -drop_count)[-drop_count:]
                keep_mask = np.ones(len(errors), dtype=bool)
                keep_mask[drop_idx] = False
                dropped_noisy = int(len(keep_mask) - keep_mask.sum())
                df = df.loc[keep_mask].reset_index(drop=True)
                logging.info(
                    "Dropped %d noisy samples via auxiliary model", dropped_noisy
                )
        else:
            drop_idx = np.argpartition(errors, -drop_count)[-drop_count:]
            keep_mask = np.ones(len(errors), dtype=bool)
            keep_mask[drop_idx] = False
            dropped_noisy = int(len(keep_mask) - keep_mask.sum())
            df = df.loc[keep_mask].reset_index(drop=True)
            logging.info(
                "Dropped %d noisy samples via auxiliary model", dropped_noisy
            )
        # re-normalise weights after filtering
        sw = df["sample_weight"].to_numpy(dtype=float)
        mean_sw = float(np.mean(sw))
        if mean_sw > 0:
            df["sample_weight"] = sw / mean_sw

    if use_autoencoder and feature_names:
        X_raw = df[feature_names].to_numpy(dtype=float)
        ae_path = out_dir / "autoencoder.pt"
        _train_autoencoder_weights(
            X_raw,
            ae_path,
            latent_dim=autoencoder_dim,
            epochs=autoencoder_epochs,
            device=device,
        )
        embedded = _encode_with_autoencoder(X_raw, ae_path, device=device)
        df.drop(columns=feature_names, inplace=True)
        ae_cols = [f"ae_{i}" for i in range(embedded.shape[1])]
        for i, col in enumerate(ae_cols):
            df[col] = embedded[:, i]
        feature_names = ae_cols

    label_cols = [c for c in df.columns if c.startswith("label")]
    active_labels = [c for c in label_cols if df[c].nunique() > 1]
    if len(active_labels) > 1:
        _train_multi_output_clf(df, feature_names, active_labels, out_dir)
        return

    optuna_info: dict[str, object] | None = None
    if optuna_trials > 0 and _HAS_OPTUNA:
        X_all = df[feature_names].to_numpy()
        y_all = df["label"].astype(int).to_numpy()
        sw_all = df["sample_weight"].to_numpy()
        profit_col = (
            "profit"
            if "profit" in df.columns
            else ("net_profit" if "net_profit" in df.columns else None)
        )

        def objective(trial: "optuna.Trial") -> float:
            model_type = trial.suggest_categorical("model_type", ["sgd", "gboost"])
            threshold = trial.suggest_float("threshold", 0.1, 0.9)
            if model_type == "sgd":
                lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
                C = trial.suggest_float("C", logreg_c_low, logreg_c_high, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", logreg_l1_low, logreg_l1_high)
            else:
                lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                n_estimators = trial.suggest_int(
                    "n_estimators", gboost_n_estimators_low, gboost_n_estimators_high
                )
                subsample = trial.suggest_float(
                    "subsample", gboost_subsample_low, gboost_subsample_high
                )
            outer = PurgedWalkForward(
                n_splits=min(optuna_folds, len(X_all) - 1), gap=purge_gap
            )
            acc_scores: list[float] = []
            profit_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(outer.split(X_all)):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]
                w_train = sw_all[train_idx]
                if model_type == "sgd":
                    X_train, c_low, c_high = _clip_train_features(X_train)
                    X_val = _clip_apply(X_val, c_low, c_high)
                    scaler = RobustScaler().fit(X_train)
                    class_counts = np.bincount(y_train)
                    val_size = int(len(y_train) * 0.1)
                    if len(class_counts) < 2 or class_counts.min() < 2 or val_size < 2:
                        clf = SGDClassifier(
                            loss="log_loss",
                            learning_rate="constant",
                            eta0=lr,
                            alpha=1.0 / C,
                            penalty="elasticnet",
                            l1_ratio=l1_ratio,
                        )
                        clf.partial_fit(
                            scaler.transform(X_train),
                            y_train,
                            classes=np.array([0, 1]),
                            sample_weight=w_train,
                        )
                    else:
                        clf = SGDClassifier(
                            loss="log_loss",
                            learning_rate="constant",
                            eta0=lr,
                            alpha=1.0 / C,
                            penalty="elasticnet",
                            l1_ratio=l1_ratio,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=5,
                        )
                        clf.fit(
                            scaler.transform(X_train),
                            y_train,
                            sample_weight=w_train,
                        )
                    probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
                else:
                    clf = GradientBoostingClassifier(
                        learning_rate=lr,
                        max_depth=depth,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                    )
                    clf.fit(X_train, y_train, sample_weight=w_train)
                    probs = clf.predict_proba(X_val)[:, 1]
                preds = (probs >= threshold).astype(int)
                acc = accuracy_score(y_val, preds)
                acc_scores.append(acc)
                if profit_col:
                    profits_val = df.iloc[val_idx][profit_col].to_numpy()
                    profit = float((profits_val * preds).mean())
                else:
                    profit = 0.0
                profit_scores.append(profit)
                trial.report(acc, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr(
                "fold_scores",
                [
                    {"accuracy": float(a), "profit": float(p)}
                    for a, p in zip(acc_scores, profit_scores)
                ],
            )
            return float(np.mean(acc_scores)) if acc_scores else 0.0

        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=optuna_trials)
        optuna_info = {
            "params": study.best_params,
            "score": float(study.best_value),
            "fold_scores": study.best_trial.user_attrs.get("fold_scores", []),
        }

    def _session_from_hour(hour: int) -> str:
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 16:
            return "london"
        return "newyork"

    if per_regime and "regime" in df.columns:
        group_field = "regime"
        group_iter = df.groupby("regime")
    else:
        hours = df["hour"] if "hour" in df.columns else pd.Series([0] * len(df))
        df["session"] = hours.astype(int).apply(_session_from_hour)
        group_field = "session"
        group_iter = df.groupby("session")

    session_models: dict[str, dict[str, object]] = {}
    cv_acc_all: list[float] = []
    cv_profit_all: list[float] = []
    for name, group in group_iter:
        model_name = f"regime_{int(name)}" if group_field == "regime" else name
        if len(group) < 2:
            continue
        feat_cols = [c for c in feature_names if c != group_field]
        X_all = group[feat_cols].to_numpy()
        y_all = group["label"].astype(int).to_numpy()
        w_all = group["sample_weight"].to_numpy()
        n_splits = min(5, len(group) - 1)
        if n_splits < 1:
            continue
        if n_splits < 2:
            splits = [
                (
                    np.arange(len(group) - 1),
                    np.arange(len(group) - 1, len(group)),
                )
            ]
        else:
            tscv = PurgedWalkForward(n_splits=n_splits, gap=purge_gap)
            splits = list(tscv.split(X_all))

        threshold_grid = np.linspace(0.0, 1.0, 101)
        profits_matrix: list[list[float]] = []
        acc_matrix: list[list[float]] = []
        rec_matrix: list[list[float]] = []
        # Store probabilities and labels from validation folds for conformal bounds
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        profit_col = (
            "profit"
            if "profit" in group.columns
            else ("net_profit" if "net_profit" in group.columns else None)
        )
        for train_idx, val_idx in splits:
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            w_train = w_all[train_idx]
            X_train, c_low, c_high = _clip_train_features(X_train)
            X_val = _clip_apply(X_val, c_low, c_high)
            scaler = RobustScaler().fit(X_train)
            class_counts = np.bincount(y_train)
            val_size = int(len(y_train) * 0.1)
            if len(class_counts) < 2 or class_counts.min() < 2 or val_size < 2:
                clf = SGDClassifier(loss="log_loss")
                clf.partial_fit(
                    scaler.transform(X_train),
                    y_train,
                    classes=np.array([0, 1]),
                    sample_weight=w_train,
                )
            else:
                clf = SGDClassifier(
                    loss="log_loss",
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                )
                clf.fit(
                    scaler.transform(X_train),
                    y_train,
                    sample_weight=w_train,
                )
            probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
            # collect probabilities and labels for conformal interval computation
            all_probs.append(probs)
            all_labels.append(y_val)
            profits_val = (
                group.iloc[val_idx][profit_col].to_numpy()
                if profit_col
                else np.zeros_like(y_val, dtype=float)
            )
            fold_profits: list[float] = []
            fold_accs: list[float] = []
            fold_recs: list[float] = []
            for t in threshold_grid:
                preds = (probs >= t).astype(int)
                fold_accs.append(accuracy_score(y_val, preds))
                fold_recs.append(recall_score(y_val, preds, zero_division=0))
                profit = (
                    float((profits_val * preds).mean()) if len(profits_val) else 0.0
                )
                fold_profits.append(profit)
            profits_matrix.append(fold_profits)
            acc_matrix.append(fold_accs)
            rec_matrix.append(fold_recs)
        profits_arr = np.asarray(profits_matrix)
        acc_arr = np.asarray(acc_matrix)
        rec_arr = np.asarray(rec_matrix)
        mean_profit_by_thresh = profits_arr.mean(axis=0)
        best_idx = int(np.argmax(mean_profit_by_thresh))
        best_thresh = float(threshold_grid[best_idx])
        fold_metrics = [
            {
                "accuracy": float(acc_arr[i, best_idx]),
                "recall": float(rec_arr[i, best_idx]),
                "profit": float(profits_arr[i, best_idx]),
            }
            for i in range(len(profits_matrix))
        ]
        if not any(
            fm["accuracy"] >= min_accuracy or fm["profit"] >= min_profit
            for fm in fold_metrics
        ):
            raise ValueError(
                f"Session {name} failed to meet min accuracy {min_accuracy} or profit {min_profit}"
            )
        mean_acc = float(acc_arr[:, best_idx].mean()) if len(acc_arr) else 0.0
        mean_rec = float(rec_arr[:, best_idx].mean()) if len(rec_arr) else 0.0
        mean_profit = (
            float(mean_profit_by_thresh[best_idx]) if len(mean_profit_by_thresh) else 0.0
        )
        avg_thresh = best_thresh
        # Compute conformal bounds using validation probabilities
        probs_flat = np.concatenate(all_probs) if all_probs else np.array([])
        labels_flat = np.concatenate(all_labels) if all_labels else np.array([])
        pos_probs = probs_flat[labels_flat == 1]
        neg_probs = probs_flat[labels_flat == 0]
        alpha = 0.05
        conf_lower = float(np.quantile(neg_probs, 1 - alpha)) if len(neg_probs) else 0.0
        conf_upper = float(np.quantile(pos_probs, alpha)) if len(pos_probs) else 1.0
        if conf_lower > conf_upper:
            conf_lower, conf_upper = conf_upper, conf_lower
        X_all_clip, _, _ = _clip_train_features(X_all)
        scaler_full = RobustScaler().fit(X_all_clip)
        class_counts_full = np.bincount(y_all)
        val_size_full = int(len(y_all) * 0.1)
        X_scaled_full = scaler_full.transform(X_all_clip)
        if len(class_counts_full) < 2 or class_counts_full.min() < 2 or val_size_full < 2:
            clf_full = SGDClassifier(loss="log_loss")
            clf_full.partial_fit(
                X_scaled_full,
                y_all,
                classes=np.array([0, 1]),
                sample_weight=w_all,
            )
        else:
            clf_full = SGDClassifier(
                loss="log_loss",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
            )
            clf_full.fit(
                X_scaled_full,
                y_all,
                sample_weight=w_all,
            )

        def _fit_regression(target_names: list[str]) -> dict | None:
            for col in target_names:
                if col in group.columns:
                    y = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
                    reg = LinearRegression().fit(X_scaled_full, y, sample_weight=w_all)
                    return {
                        "coefficients": reg.coef_.astype(float).tolist(),
                        "intercept": float(reg.intercept_),
                    }
            return None

        lot_model = _fit_regression(["lot", "lot_size", "lots"])
        sl_model = _fit_regression(
            [
                "sl_distance",
                "stop_loss_distance",
                "stop_loss",
            ]
        )
        tp_model = _fit_regression(
            [
                "tp_distance",
                "take_profit_distance",
                "take_profit",
            ]
        )

        params: dict[str, object] = {
            "coefficients": clf_full.coef_[0].astype(float).tolist(),
            "intercept": float(clf_full.intercept_[0]),
            "threshold": avg_thresh,
            "feature_mean": scaler_full.center_.astype(float).tolist(),
            "feature_std": scaler_full.scale_.astype(float).tolist(),
            "metrics": {
                "accuracy": mean_acc,
                "recall": mean_rec,
                "profit": mean_profit,
            },
            "cv_metrics": fold_metrics,
            "conformal_lower": conf_lower,
            "conformal_upper": conf_upper,
        }
        if unc_df is not None and name == uncertain_session:
            uX = unc_df[feature_names].to_numpy(dtype=float)
            uy = pd.to_numeric(unc_df["label"], errors="coerce").to_numpy(dtype=float)
            X_scaled_u = scaler_full.transform(uX)
            probs_u = clf_full.predict_proba(X_scaled_u)[:,1]
            preds_u = (probs_u >= avg_thresh).astype(int)
            acc_u = accuracy_score(uy, preds_u)
            rec_u = recall_score(uy, preds_u, zero_division=0)
            logging.info(
                "uncertain sample metrics: accuracy=%.3f recall=%.3f n=%d",
                acc_u,
                rec_u,
                len(uy),
            )
            params["metrics"]["uncertain_accuracy"] = float(acc_u)
            params["metrics"]["uncertain_recall"] = float(rec_u)
        if lot_model:
            params["lot_model"] = lot_model
        if sl_model:
            params["sl_model"] = sl_model
        if tp_model:
            params["tp_model"] = tp_model
        params["n_samples"] = int(len(group))
        session_models[model_name] = params
        cv_acc_all.append(mean_acc)
        cv_profit_all.append(mean_profit)

    if not session_models:
        raise ValueError(f"No training data found in {data_dir}")

    symbol_thresholds: dict[str, float] = {}
    if "symbol" in df.columns:
        feat_cols_all = [c for c in feature_names if c != "session"]
        for sym, sym_df in df.groupby("symbol"):
            if len(sym_df) < 2:
                continue
            X_sym = sym_df[feat_cols_all].to_numpy()
            y_sym = sym_df["label"].astype(int).to_numpy()
            w_sym = sym_df["sample_weight"].to_numpy()
            X_sym, _, _ = _clip_train_features(X_sym)
            scaler_sym = RobustScaler().fit(X_sym)
            class_counts = np.bincount(y_sym)
            val_size_sym = int(len(y_sym) * 0.1)
            if len(class_counts) < 2 or class_counts.min() < 2 or val_size_sym < 2:
                base_clf = SGDClassifier(loss="log_loss")
                base_clf.partial_fit(
                    scaler_sym.transform(X_sym),
                    y_sym,
                    classes=np.array([0, 1]),
                    sample_weight=w_sym,
                )
            else:
                base_clf = SGDClassifier(
                    loss="log_loss",
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                )
                base_clf.fit(
                    scaler_sym.transform(X_sym),
                    y_sym,
                    sample_weight=w_sym,
                )
            calib = CalibratedClassifierCV(base_clf, method="isotonic", cv="prefit")
            calib.fit(
                scaler_sym.transform(X_sym),
                y_sym,
                sample_weight=w_sym,
            )
            probs = calib.predict_proba(scaler_sym.transform(X_sym))[:, 1]
            thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
            best_thresh = 0.5
            best_acc = -1.0
            profit_col = (
                "profit"
                if "profit" in sym_df.columns
                else ("net_profit" if "net_profit" in sym_df.columns else None)
            )
            profits = (
                sym_df[profit_col].to_numpy()
                if profit_col
                else np.zeros_like(y_sym, dtype=float)
            )
            for t in thresholds:
                preds = (probs >= t).astype(int)
                acc = accuracy_score(y_sym, preds)
                profit = float((profits * preds).mean()) if len(profits) else 0.0
                if acc > best_acc:
                    best_acc = float(acc)
                    best_thresh = float(t)
            symbol_thresholds[str(sym)] = best_thresh

    # Aggregate conformal bounds across sessions for convenience
    conf_lowers = [p["conformal_lower"] for p in session_models.values()]
    conf_uppers = [p["conformal_upper"] for p in session_models.values()]
    gating_params = None
    if per_regime and "regime" in df.columns:
        regime_cols = [c for c in df.columns if c.startswith("regime_")]
        if regime_cols:
            gating_params = train_meta_model(df, regime_cols, label_col="regime")
    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "half_life_days": float(half_life_days),
        "session_models": session_models,
        "training_mode": "lite",
        "mode": mode,
        "cv_accuracy": float(np.mean(cv_acc_all)) if cv_acc_all else 0.0,
        "cv_profit": float(np.mean(cv_profit_all)) if cv_profit_all else 0.0,
        "conformal_lower": float(min(conf_lowers)) if conf_lowers else 0.0,
        "conformal_upper": float(max(conf_uppers)) if conf_uppers else 1.0,
        "training_rows": int(len(df)),
        "dropped_noisy": int(dropped_noisy),
    }
    model["meta_barriers"] = {
        "take_profit_mult": float(take_profit_mult),
        "stop_loss_mult": float(stop_loss_mult),
        "hold_period": int(hold_period),
        "use_meta_label": bool(use_meta_label),
    }
    if meta_init is not None:
        model["meta_weights"] = meta_init.astype(float).tolist()
    w_stats = df["sample_weight"].to_numpy(dtype=float)
    model["sample_weight_stats"] = {
        "mean": float(w_stats.mean()) if len(w_stats) else 0.0,
        "min": float(w_stats.min()) if len(w_stats) else 0.0,
        "max": float(w_stats.max()) if len(w_stats) else 0.0,
    }
    profit_col = "profit" if "profit" in df.columns else (
        "net_profit" if "net_profit" in df.columns else None
    )
    model["weighted_by_profit"] = bool(use_profit_weight and profit_col is not None)
    if use_autoencoder:
        model["autoencoder"] = "autoencoder.pt"
    if tick_encoder is not None:
        _persist_encoder_meta(out_dir, model, tick_encoder)
    if not per_regime:
        model["session_hours"] = {
            "asian": [0, 8],
            "london": [8, 16],
            "newyork": [16, 24],
        }
    if symbol_thresholds:
        model["symbol_thresholds"] = symbol_thresholds
    if gating_params:
        model["regime_gating"] = gating_params

    # ------------------------------------------------------------------
    # Train alternative base models and a simple gating network
    # ------------------------------------------------------------------
    def _fit_base_model(X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Fit logistic regression and return params, predictor and model."""
        if smote_threshold is not None:
            X, y, w = _maybe_smote(X, y, w, threshold=smote_threshold)
        X_clip, c_low, c_high = _clip_train_features(X)
        scaler = RobustScaler().fit(X_clip)
        X_scaled = scaler.transform(X_clip)

        clf: object
        coef: np.ndarray
        intercept: float

        if meta_init is not None and len(meta_init) == X_scaled.shape[1]:
            w_vec = np.concatenate([meta_init.astype(float), np.zeros(1)])
            Xb = np.hstack([X_scaled, np.ones((len(X_scaled), 1))])
            for _ in range(100):
                preds = _sigmoid(Xb @ w_vec)
                grad = Xb.T @ (preds - y) / len(y)
                w_vec -= 0.1 * grad
            coef = w_vec[:-1]
            intercept = float(w_vec[-1])

            class _GDClf:
                def __init__(self, c: np.ndarray, i: float) -> None:
                    self.coef_ = np.array([c], dtype=float)
                    self.intercept_ = np.array([i], dtype=float)

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    z = X @ self.coef_[0] + self.intercept_[0]
                    p = _sigmoid(z)
                    return np.vstack([1 - p, p]).T

            clf = _GDClf(coef, intercept)
        else:
            class_counts = np.bincount(y)
            val_size = int(len(y) * 0.1)
            if len(class_counts) < 2 or class_counts.min() < 2 or val_size < 2:
                clf = SGDClassifier(loss="log_loss")
                clf.partial_fit(
                    X_scaled,
                    y,
                    classes=np.array([0, 1]),
                    sample_weight=w,
                )
            else:
                clf = SGDClassifier(
                    loss="log_loss",
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                )
                clf.fit(
                    X_scaled,
                    y,
                    sample_weight=w,
                )
            coef = clf.coef_[0]
            intercept = float(clf.intercept_[0])

        def _predict(inp: np.ndarray) -> np.ndarray:
            return clf.predict_proba(
                scaler.transform(_clip_apply(inp, c_low, c_high))
            )[:, 1]

        params = {
            "coefficients": coef.astype(float).tolist(),
            "intercept": float(intercept),
            "threshold": 0.5,
            "feature_mean": scaler.center_.astype(float).tolist(),
            "feature_std": scaler.scale_.astype(float).tolist(),
            "conformal_lower": 0.0,
            "conformal_upper": 1.0,
        }
        return params, _predict, clf, scaler, c_low, c_high

    # Use spread as a crude volatility proxy; fall back to zeros
    vol_series = pd.to_numeric(
        df.get("spread", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    hours_series = pd.to_numeric(
        df.get("hour", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0)
    feat_cols = [c for c in feature_names if c != "session"]
    X_all = df[feat_cols].to_numpy()
    y_all = df["label"].astype(int).to_numpy()
    w_all = df["sample_weight"].to_numpy()

    median_vol = float(np.median(vol_series.to_numpy())) if len(vol_series) else 0.0
    high_mask = vol_series.to_numpy() > median_vol
    low_mask = ~high_mask

    # Fit model and compute SHAP importances for optional pruning
    (
        params_generic,
        pred_generic,
        clf_generic,
        scaler_generic,
        c_low_gen,
        c_high_gen,
    ) = _fit_base_model(X_all, y_all, w_all)
    mean_abs_shap = _mean_abs_shap_linear(
        clf_generic, scaler_generic, _clip_apply(X_all, c_low_gen, c_high_gen)
    )

    keep_mask = mean_abs_shap >= prune_threshold
    if prune_threshold > 0.0 and not keep_mask.all():
        feat_cols = [n for n, k in zip(feat_cols, keep_mask) if k]
        X_all = X_all[:, keep_mask]
        (
            params_generic,
            pred_generic,
            clf_generic,
            scaler_generic,
            c_low_gen,
            c_high_gen,
        ) = _fit_base_model(X_all, y_all, w_all)
        mean_abs_shap = _mean_abs_shap_linear(
            clf_generic, scaler_generic, _clip_apply(X_all, c_low_gen, c_high_gen)
        )

    feature_names = feat_cols

    models: dict[str, dict[str, object]] = {}
    pred_funcs: list[Callable[[np.ndarray], np.ndarray]] = []
    models["logreg"] = params_generic
    pred_funcs.append(pred_generic)

    # Compute feature importance for the final model
    ranked_feats = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    feature_importance = {n: float(v) for n, v in ranked_feats}
    logging.info("Ranked feature importances: %s", ranked_feats)

    if high_mask.sum() >= 2:
        params_high, pred_high, _, _, _, _ = _fit_base_model(
            X_all[high_mask], y_all[high_mask], w_all[high_mask]
        )
    else:
        params_high, pred_high = params_generic, pred_generic
    models["xgboost"] = params_high
    pred_funcs.append(pred_high)

    if low_mask.sum() >= 2:
        params_low, pred_low, _, _, _, _ = _fit_base_model(
            X_all[low_mask], y_all[low_mask], w_all[low_mask]
        )
    else:
        params_low, pred_low = params_generic, pred_generic
    models["lstm"] = params_low
    pred_funcs.append(pred_low)

    model["models"] = models
    model["feature_importance"] = feature_importance
    model["retained_features"] = feature_names
    model["feature_names"] = feature_names

    # Determine which model performs best per sample
    probs = np.vstack([f(X_all) for f in pred_funcs])
    errors = np.abs(probs - y_all)
    best_idx = np.argmin(errors, axis=0)

    router_feats = np.column_stack([vol_series.to_numpy(), hours_series.to_numpy()])
    router_feats, r_low, r_high = _clip_train_features(router_feats)
    scaler_router = RobustScaler().fit(router_feats)
    norm_router = scaler_router.transform(router_feats)
    class_counts = np.bincount(best_idx)
    val_size_router = int(len(best_idx) * 0.1)
    if len(class_counts) < 2 or class_counts.min() < 2 or val_size_router < 2:
        router_clf = SGDClassifier(loss="log_loss")
        router_clf.partial_fit(
            norm_router, best_idx, classes=np.array([0, 1, 2]), sample_weight=w_all
        )
    else:
        router_clf = SGDClassifier(
            loss="log_loss",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )
        router_clf.fit(norm_router, best_idx, sample_weight=w_all)
    router = {
        "intercept": router_clf.intercept_.astype(float).tolist(),
        "coefficients": router_clf.coef_.astype(float).tolist(),
       "feature_mean": scaler_router.center_.astype(float).tolist(),
        "feature_std": scaler_router.scale_.astype(float).tolist(),
    }
    model["ensemble_router"] = router

    if bayesian_ensembles and bayesian_ensembles > 1:
        X_be, c_low_be, c_high_be = _clip_train_features(X_all)
        scaler_be = RobustScaler().fit(X_be)
        X_scaled_be = scaler_be.transform(X_be)
        prob_list: list[np.ndarray] = []
        for seed in range(int(bayesian_ensembles)):
            clf_be = SGDClassifier(loss="log_loss", random_state=seed)
            clf_be.partial_fit(
                X_scaled_be,
                y_all,
                classes=np.array([0, 1]),
                sample_weight=w_all,
            )
            probs = clf_be.predict_proba(X_scaled_be)[:, 1]
            prob_list.append(probs)
        prob_arr = np.vstack(prob_list)
        avg_probs = prob_arr.mean(axis=0)
        var_probs = float(prob_arr.var(axis=0).mean())
        single_probs = prob_list[0]
        if len(np.unique(y_all)) > 1:
            roc_single = float(roc_auc_score(y_all, single_probs))
            roc_ens = float(roc_auc_score(y_all, avg_probs))
        else:
            roc_single = roc_ens = 0.0
        brier_single = float(brier_score_loss(y_all, single_probs))
        brier_ens = float(brier_score_loss(y_all, avg_probs))
        model["bayesian_ensemble"] = {
            "size": int(bayesian_ensembles),
            "variance": var_probs,
            "metrics": {
                "single": {"roc_auc": roc_single, "brier": brier_single},
                "ensemble": {"roc_auc": roc_ens, "brier": brier_ens},
            },
        }

    if ensemble in {"voting", "stacking"}:
        base_estimators = [
            (
                "logreg",
                make_pipeline(
                    RobustScaler(), LogisticRegression(max_iter=1000, random_state=0)
                ),
            ),
            (
                "gboost",
                GradientBoostingClassifier(
                    random_state=0,
                    n_estimators=5,
                    max_depth=1,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                ),
            ),
        ]
        if ensemble == "voting":
            ensemble_clf = VotingClassifier(estimators=base_estimators, voting="soft")
        else:
            ensemble_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=0),
            )
        X_all_ens, _, _ = _clip_train_features(X_all)
        set_config(enable_metadata_routing=True)
        try:
            ensemble_clf.set_fit_request(sample_weight=True)
            for _, est in ensemble_clf.estimators:
                try:
                    est.set_fit_request(sample_weight=True)
                except Exception:
                    pass
            if ensemble == "stacking":
                try:
                    ensemble_clf.final_estimator_.set_fit_request(sample_weight=True)
                except Exception:
                    pass
            ensemble_clf.fit(X_all_ens, y_all, sample_weight=w_all)
        except Exception:
            ensemble_clf.fit(X_all_ens, y_all)
        base_acc = {
            name: float(accuracy_score(y_all, est.predict(X_all_ens)))
            for name, est in ensemble_clf.named_estimators_.items()
        }
        ensemble_acc = float(accuracy_score(y_all, ensemble_clf.predict(X_all_ens)))
        if ensemble == "voting":
            weights_out = (
                list(ensemble_clf.weights)
                if ensemble_clf.weights is not None
                else [1.0] * len(base_estimators)
            )
        else:
            weights_out = ensemble_clf.final_estimator_.coef_.ravel().tolist()
        model["ensemble"] = {
            "type": ensemble,
            "estimators": [name for name, _ in base_estimators],
            "weights": [float(w) for w in weights_out],
            "accuracy": ensemble_acc,
            "base_accuracies": base_acc,
        }
    if embeddings:
        model["symbol_embeddings"] = embeddings
    if gnn_state:
        model["gnn_state"] = gnn_state
    if optuna_info:
        model["optuna_best_params"] = optuna_info.get("params", {})
        model["optuna_best_score"] = optuna_info.get("score", 0.0)
        if optuna_info.get("fold_scores"):
            model["optuna_best_fold_scores"] = optuna_info.get("fold_scores", [])
    synth_count = int(df["synthetic"].sum()) if "synthetic" in df.columns else 0
    synthetic_info = {
        "real": int(len(df) - synth_count),
        "synthetic": synth_count,
        "all": int(len(df)),
        "synthetic_fraction": float(synth_count / len(df)) if len(df) else 0.0,
        "synthetic_weight": float(synthetic_weight),
    }
    model["synthetic_metrics"] = synthetic_info
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)
    logging.info(
        "Trained logreg model - cv_accuracy %.3f cv_profit %.3f params %s",
        model.get("cv_accuracy", 0.0),
        model.get("cv_profit", 0.0),
        model.get("optuna_best_params", {}),
    )


def _train_transformer(
    data_dir: Path,
    out_dir: Path,
    *,
    window: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    dropout: float = 0.0,
    device: str = "cpu",
    calendar_file: Path | None = None,
    symbol_graph: dict | str | Path | None = None,
    news_sentiment: pd.DataFrame | None = None,
    synthetic_model: Path | None = None,
    synthetic_frac: float = 0.0,
    synthetic_weight: float = 0.2,
    uncertain_file: Path | None = None,
    uncertain_weight: float = 2.0,
    neighbor_corr_windows: Iterable[int] | None = None,
    tick_encoder: Path | None = None,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 0.0,
    drift_weight: float = 0.0,
    bayesian_ensembles: int = 0,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    use_meta_label: bool = False,
    **_,
) -> torch.nn.Module:
    """Train a tiny attention encoder on rolling feature windows."""
    if not _HAS_TORCH:  # pragma: no cover - requires optional dependency
        raise ImportError("PyTorch is required for transformer model")

    df, feature_names, _ = _load_logs(
        data_dir,
        take_profit_mult=take_profit_mult,
        stop_loss_mult=stop_loss_mult,
        hold_period=hold_period,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    if use_meta_label and "meta_label" in df.columns:
        df["label"] = df["meta_label"]
    if "label" not in df.columns:
        raise ValueError("label column missing from data")
    df, feature_names, _, _ = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
        neighbor_corr_windows=neighbor_corr_windows,
        tick_encoder=tick_encoder,
    )

    _apply_drift_pruning(
        df, feature_names, drift_scores, drift_threshold, drift_weight
    )
    unc_df = None
    uncertain_idx = np.zeros(len(df), dtype=bool)
    if uncertain_file:
        try:
            unc_df = pd.read_csv(uncertain_file, sep=";")
            unc_df.columns = [c.lower() for c in unc_df.columns]
            if "features" in unc_df.columns:
                feats = unc_df["features"].str.split(":")
                feat_df = pd.DataFrame(feats.tolist(), columns=feature_names)
                unc_df = pd.concat([unc_df, feat_df], axis=1)
            for col in df.columns:
                if col not in unc_df.columns:
                    unc_df[col] = 0.0
            unc_df = unc_df[df.columns]
            df = pd.concat([df, unc_df], ignore_index=True)
            uncertain_idx = np.zeros(len(df), dtype=bool)
            uncertain_idx[-len(unc_df):] = True
        except Exception:
            unc_df = None

    X_all = df[feature_names].to_numpy(dtype=float)
    y_all = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    X_all, _, _ = _clip_train_features(X_all)
    scaler_t = RobustScaler().fit(X_all)
    norm_X = scaler_t.transform(X_all)
    feat_mean = scaler_t.center_
    feat_std = scaler_t.scale_

    seqs: list[list[list[float]]] = []
    ys: list[float] = []
    synth_flags: list[float] = []
    unc_flags: list[float] = []
    for i in range(window, len(norm_X)):
        seqs.append(norm_X[i - window : i].tolist())
        ys.append(y_all[i])
        synth_flags.append(0.0)
        unc_flags.append(1.0 if uncertain_idx[i] else 0.0)

    synth_last_feats = np.empty((0, len(feature_names)))
    if synthetic_model is not None and synthetic_frac > 0:
        try:
            from scripts.train_price_gan import sample_sequences

            n_real = len(seqs)
            n_synth = max(1, int(n_real * synthetic_frac))
            synth = sample_sequences(Path(synthetic_model), n_synth)
            n_feat = len(feature_names)
            expected = window * n_feat
            if synth.shape[1] != expected:
                synth = np.resize(synth, (n_synth, expected))
            synth = synth.reshape(n_synth, window, n_feat)
            synth_last_feats = synth[:, -1, :]
            p = float(y_all.mean()) if len(y_all) else 0.5
            synth_y = np.random.binomial(1, p, size=n_synth).astype(float)
            for seq, label in zip(synth, synth_y):
                seqs.append(seq.tolist())
                ys.append(float(label))
                synth_flags.append(1.0)
                unc_flags.append(0.0)
        except Exception:
            pass
    weights_arr = np.ones(len(seqs), dtype=float)
    weights_arr = np.where(np.array(synth_flags) > 0.0, synthetic_weight, weights_arr)
    weights_arr = np.where(np.array(unc_flags) > 0.0, uncertain_weight, weights_arr)
    # keep tensors on CPU for the DataLoader and move to device during training
    X = torch.tensor(seqs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    w = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(-1)
    flags = torch.tensor(synth_flags, dtype=torch.float32).unsqueeze(-1)
    ds = torch.utils.data.TensorDataset(X, y, w, flags)
    split = max(1, int(len(ds) * 0.8))
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, len(ds) - split])
    dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32)

    class TinyTransformer(torch.nn.Module):
        def __init__(self, in_dim: int, win: int, dim: int = 16, drop: float = 0.0):
            super().__init__()
            self.embed = torch.nn.Linear(in_dim, dim)
            self.pos_embed = torch.nn.Embedding(win, dim)
            self.q = torch.nn.Linear(dim, dim)
            self.k = torch.nn.Linear(dim, dim)
            self.v = torch.nn.Linear(dim, dim)
            self.attn_dropout = torch.nn.Dropout(drop)
            self.out = torch.nn.Linear(dim, 1)
            self.out_dropout = torch.nn.Dropout(drop)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F)
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            emb = self.embed(x) + self.pos_embed(positions)
            q = self.q(emb)
            k = self.k(emb)
            v = self.v(emb)
            attn = torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1
            )
            ctx = torch.matmul(attn, v).mean(dim=1)
            ctx = self.attn_dropout(ctx)
            return self.out_dropout(self.out(ctx))

    dim = 16
    dev = torch.device(device)
    model = TinyTransformer(len(feature_names), window, dim, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    use_cuda = torch.cuda.is_available() and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    best_loss = float("inf")
    best_state = None
    patience = 2
    no_improve = 0
    for _ in range(epochs):  # pragma: no cover - simple training loop
        model.train()
        for batch_x, batch_y, batch_w, _ in dl:
            batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)
            batch_w = batch_w.to(dev)
            opt.zero_grad()
            if use_cuda:
                with torch.cuda.amp.autocast():
                    logits = model(batch_x)
                    loss = (loss_fn(logits, batch_y) * batch_w).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = (loss_fn(logits, batch_y) * batch_w).mean()
                loss.backward()
                opt.step()
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy, vw, _ in val_dl:
                vx = vx.to(dev)
                vy = vy.to(dev)
                vw = vw.to(dev)
                logits = model(vx)
                val_loss += (loss_fn(logits, vy) * vw).mean().item()
        val_loss /= max(1, len(val_dl))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def _tensor_list(t: torch.Tensor) -> list:
        return t.detach().cpu().numpy().tolist()

    weights = {
        "embed_weight": _tensor_list(model.embed.weight),
        "embed_bias": _tensor_list(model.embed.bias),
        "pos_embed_weight": _tensor_list(model.pos_embed.weight),
        "q_weight": _tensor_list(model.q.weight),
        "q_bias": _tensor_list(model.q.bias),
        "k_weight": _tensor_list(model.k.weight),
        "k_bias": _tensor_list(model.k.bias),
        "v_weight": _tensor_list(model.v.weight),
        "v_bias": _tensor_list(model.v.bias),
        "out_weight": _tensor_list(model.out.weight.squeeze(0)),
        "out_bias": _tensor_list(model.out.bias),
    }

    # Distil transformer into a logistic regression student
    model.eval()
    with torch.no_grad():
        teacher_logits = model(X.to(dev)).squeeze(-1)
        teacher_probs = torch.sigmoid(teacher_logits).cpu().numpy()

    bayes_info: dict | None = None
    if bayesian_ensembles and bayesian_ensembles > 1 and dropout > 0.0:
        model.train()
        probs_list = []
        for seed in range(int(bayesian_ensembles)):
            torch.manual_seed(seed)
            with torch.no_grad():
                logits_mc = model(X.to(dev)).squeeze(-1)
                probs_list.append(torch.sigmoid(logits_mc).cpu().numpy())
        prob_arr = np.vstack(probs_list)
        avg_probs = prob_arr.mean(axis=0)
        var_probs = float(prob_arr.var(axis=0).mean())
        y_arr_full = np.array(ys)
        if len(np.unique(y_arr_full)) > 1:
            roc_single = float(roc_auc_score(y_arr_full, teacher_probs))
            roc_ens = float(roc_auc_score(y_arr_full, avg_probs))
        else:
            roc_single = roc_ens = 0.0
        brier_single = float(brier_score_loss(y_arr_full, teacher_probs))
        brier_ens = float(brier_score_loss(y_arr_full, avg_probs))
        bayes_info = {
            "size": int(bayesian_ensembles),
            "variance": var_probs,
            "metrics": {
                "single": {"roc_auc": roc_single, "brier": brier_single},
                "ensemble": {"roc_auc": roc_ens, "brier": brier_ens},
            },
        }
        model.eval()

    pred_labels = (teacher_probs > 0.5).astype(int)
    y_arr = np.array(ys)
    flags_arr = np.array(synth_flags, dtype=bool)
    unc_mask = np.array(unc_flags, dtype=bool)
    metrics_all = {
        "accuracy": float(accuracy_score(y_arr, pred_labels)),
        "recall": float(recall_score(y_arr, pred_labels, zero_division=0)),
    }
    if unc_mask.any():
        metrics_all["uncertain_accuracy"] = float(
            accuracy_score(y_arr[unc_mask], pred_labels[unc_mask])
        )
        metrics_all["uncertain_recall"] = float(
            recall_score(y_arr[unc_mask], pred_labels[unc_mask], zero_division=0)
        )
        logging.info(
            "uncertain sample metrics: accuracy=%.3f recall=%.3f n=%d",
            metrics_all["uncertain_accuracy"],
            metrics_all["uncertain_recall"],
            int(unc_mask.sum()),
        )
    real_mask = ~flags_arr
    if real_mask.any():
        metrics_real = {
            "accuracy": float(accuracy_score(y_arr[real_mask], pred_labels[real_mask])),
            "recall": float(
                recall_score(y_arr[real_mask], pred_labels[real_mask], zero_division=0)
            ),
        }
    else:
        metrics_real = {"accuracy": 0.0, "recall": 0.0}
    teacher_metrics = metrics_all
    synthetic_info = {
        "all": metrics_all,
        "real": metrics_real,
        "synthetic_fraction": float(flags_arr.mean()) if len(flags_arr) else 0.0,
        "synthetic_weight": float(synthetic_weight),
        "accuracy_delta": metrics_all["accuracy"] - metrics_real["accuracy"],
    }
    # Fit linear regression on teacher logits to approximate probabilities
    eps = 1e-6
    logits = np.log(
        teacher_probs.clip(eps, 1 - eps) / (1 - teacher_probs.clip(eps, 1 - eps))
    )
    linreg = LinearRegression()
    base_features = norm_X[window:]
    if len(synth_last_feats):
        linreg_X = np.vstack([base_features, synth_last_feats])
    else:
        linreg_X = base_features
    linreg.fit(linreg_X, logits, sample_weight=weights_arr)
    distilled = {
        "intercept": float(linreg.intercept_),
        "coefficients": [float(c) for c in linreg.coef_.tolist()],
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "threshold": 0.5,
    }

    model_json = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "transformer",
        "window_size": window,
        "feature_names": feature_names,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "weights": weights,
        "dropout": float(dropout),
        "teacher_metrics": teacher_metrics,
        "synthetic_metrics": synthetic_info,
        "distilled": distilled,
        "models": {"logreg": distilled},
    }
    model_json["meta_barriers"] = {
        "take_profit_mult": float(take_profit_mult),
        "stop_loss_mult": float(stop_loss_mult),
        "hold_period": int(hold_period),
        "use_meta_label": bool(use_meta_label),
    }
    if bayes_info:
        model_json["bayesian_ensemble"] = bayes_info
    out_dir.mkdir(parents=True, exist_ok=True)
    _persist_encoder_meta(out_dir, model_json, tick_encoder)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_json, f)

    return model


def _train_tab_transformer(
    data_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    dropout: float = 0.0,
    device: str = "cpu",
    calendar_file: Path | None = None,
    symbol_graph: dict | str | Path | None = None,
    news_sentiment: pd.DataFrame | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    tick_encoder: Path | None = None,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 0.0,
    drift_weight: float = 0.0,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    use_meta_label: bool = False,
    **_,
) -> TabTransformer:
    """Train a :class:`TabTransformer` on tabular features."""
    if not _HAS_TORCH:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for tabtransformer model")

    df, feature_names, _ = _load_logs(
        data_dir,
        take_profit_mult=take_profit_mult,
        stop_loss_mult=stop_loss_mult,
        hold_period=hold_period,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    if use_meta_label and "meta_label" in df.columns:
        df["label"] = df["meta_label"]
    if "label" not in df.columns:
        raise ValueError("label column missing from data")
    df, feature_names, _, _ = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
        neighbor_corr_windows=neighbor_corr_windows,
        tick_encoder=tick_encoder,
    )

    _apply_drift_pruning(
        df, feature_names, drift_scores, drift_threshold, drift_weight
    )

    X = df[feature_names].to_numpy(dtype=float)
    y = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    X, c_low, c_high = _clip_train_features(X)
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X)

    state, predict_fn, model = fit_tab_transformer(
        X_scaled, y, epochs=epochs, lr=lr, dropout=dropout, device=device
    )

    probs = predict_fn(X_scaled)
    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y, preds)) if len(y) else 0.0

    model_json = {
        "model_type": "tabtransformer",
        "feature_names": feature_names,
        "mean": scaler.center_.tolist(),
        "std": scaler.scale_.tolist(),
        "clip_low": c_low.tolist(),
        "clip_high": c_high.tolist(),
        "state_dict": state,
        "metrics": {"accuracy": acc},
        "meta_barriers": {
            "take_profit_mult": float(take_profit_mult),
            "stop_loss_mult": float(stop_loss_mult),
            "hold_period": int(hold_period),
            "use_meta_label": bool(use_meta_label),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_json, f)

    return model


def _train_tree_model(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str,
    min_accuracy: float = 0.0,
    min_profit: float = 0.0,
    half_life_days: float = 0.0,
    replay_file: Path | None = None,
    replay_weight: float = 1.0,
    uncertain_file: Path | None = None,
    uncertain_weight: float = 2.0,
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    news_sentiment: pd.DataFrame | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    use_volatility_weight: bool = False,
    tick_encoder: Path | None = None,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 0.0,
    drift_weight: float = 0.0,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    use_meta_label: bool = False,
    **_: object,
) -> None:
    """Train tree-based models (XGBoost, LightGBM, CatBoost)."""
    df, feature_names, _ = _load_logs(
        data_dir,
        take_profit_mult=take_profit_mult,
        stop_loss_mult=stop_loss_mult,
        hold_period=hold_period,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    feature_names = list(feature_names)
    if use_meta_label and "meta_label" in df.columns:
        df["label"] = df["meta_label"]
    if replay_file:
        rdf = pd.read_csv(replay_file)
        rdf.columns = [c.lower() for c in rdf.columns]
        df = pd.concat([df, rdf], ignore_index=True)
    unc_df = None
    if uncertain_file:
        try:
            unc_df = pd.read_csv(uncertain_file, sep=";")
            unc_df.columns = [c.lower() for c in unc_df.columns]
            if "features" in unc_df.columns:
                feats = unc_df["features"].str.split(":")
                feat_df = pd.DataFrame(feats.tolist(), columns=feature_names)
                unc_df = pd.concat([unc_df, feat_df], axis=1)
            for col in df.columns:
                if col not in unc_df.columns:
                    unc_df[col] = 0.0
            unc_df = unc_df[df.columns]
            df = pd.concat([df, unc_df], ignore_index=True)
        except Exception:
            unc_df = None
    if "net_profit" in df.columns:
        weights = pd.to_numeric(df["net_profit"], errors="coerce").abs().to_numpy()
    elif "lots" in df.columns:
        weights = pd.to_numeric(df["lots"], errors="coerce").abs().to_numpy()
    else:
        weights = np.ones(len(df), dtype=float)
    if replay_file:
        weights[-len(rdf) :] *= replay_weight
    if uncertain_file and unc_df is not None:
        weights[-len(unc_df) :] *= uncertain_weight
    if "event_time" in df.columns:
        event_times = df["event_time"].to_numpy(dtype="datetime64[s]")
        if half_life_days > 0:
            age_days, decay = _compute_decay_weights(event_times, half_life_days)
            weights = weights * decay
        else:
            ref_time = event_times.max()
            age_days = (
                (ref_time - event_times).astype("timedelta64[s]").astype(float)
                / (24 * 3600)
            )
    else:
        age_days = pd.Series(0.0, index=df.index)
        if half_life_days > 0:
            decay = 0.5 ** (age_days / half_life_days)
            weights = weights * decay
    df["age_days"] = age_days

    # Normalise weights before fitting
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w
    df["sample_weight"] = weights

    if "label" not in df.columns:
        raise ValueError("label column missing from data")

    df, feature_names, _, _ = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
        neighbor_corr_windows=neighbor_corr_windows,
        tick_encoder=tick_encoder,
    )

    _apply_drift_pruning(
        df, feature_names, drift_scores, drift_threshold, drift_weight
    )
    if use_volatility_weight and "price_volatility" in df.columns:
        vol = pd.to_numeric(df["price_volatility"], errors="coerce").fillna(0.0)
        mean_vol = float(vol.mean())
        if mean_vol > 0:
            df["sample_weight"] = df["sample_weight"] * (vol / mean_vol)
    # Re-normalise after volatility adjustment
    sw = df["sample_weight"].to_numpy(dtype=float)
    mean_sw = float(np.mean(sw))
    if mean_sw > 0:
        df["sample_weight"] = sw / mean_sw
    feature_names = [
        c
        for c in feature_names
        if c not in {"label", "profit", "net_profit", "hour", "day_of_week", "symbol"}
    ]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    w = df["sample_weight"].to_numpy(dtype=float)
    profit_col = "profit" if "profit" in df.columns else (
        "net_profit" if "net_profit" in df.columns else None
    )
    profits = df[profit_col].to_numpy(dtype=float) if profit_col else np.zeros(len(df))

    n_splits = max(1, min(5, len(X) - 1))
    tscv = PurgedWalkForward(n_splits=n_splits, gap=1)
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = w[train_idx]
    profits_val = profits[val_idx]

    if model_type == "xgboost":
        clf = fit_xgb_classifier(X_train, y_train, sample_weight=w_train)
    elif model_type == "lgbm":
        clf = fit_lgbm_classifier(X_train, y_train, sample_weight=w_train)
    elif model_type == "catboost":
        clf = fit_catboost_classifier(X_train, y_train, sample_weight=w_train)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported model type: {model_type}")

    preds = clf.predict(X_val)
    acc = float(accuracy_score(y_val, preds)) if len(y_val) else 0.0
    profit = (
        float((profits_val * preds).mean()) if len(profits_val) else 0.0
    )
    if acc < min_accuracy and profit < min_profit:
        raise ValueError(
            f"Model failed to meet min accuracy {min_accuracy} or profit {min_profit}"
        )
    acc_u = rec_u = None
    if uncertain_file and unc_df is not None and len(unc_df):
        uX = unc_df[feature_names].to_numpy(dtype=float)
        uy = pd.to_numeric(unc_df["label"], errors="coerce").to_numpy(dtype=float)
        if hasattr(clf, "predict_proba"):
            probs_u = clf.predict_proba(uX)[:, 1]
        else:
            probs_u = clf.predict(uX)
        preds_u = (probs_u >= 0.5).astype(int)
        acc_u = accuracy_score(uy, preds_u)
        rec_u = recall_score(uy, preds_u, zero_division=0)
        logging.info(
            "uncertain sample metrics: accuracy=%.3f recall=%.3f n=%d",
            acc_u,
            rec_u,
            len(uy),
        )

    mean_abs_shap = _mean_abs_shap_tree(clf, X_train)
    ranked_feats = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    logging.info("Ranked feature importances: %s", ranked_feats)

    model_json = {
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "half_life_days": float(half_life_days),
        "validation_metrics": {"accuracy": acc, "profit": profit},
        "meta_barriers": {
            "take_profit_mult": float(take_profit_mult),
            "stop_loss_mult": float(stop_loss_mult),
            "hold_period": int(hold_period),
            "use_meta_label": bool(use_meta_label),
        },
    }
    if acc_u is not None and rec_u is not None:
        vm = model_json["validation_metrics"]
        vm["uncertain_accuracy"] = float(acc_u)
        vm["uncertain_recall"] = float(rec_u)
    out_dir.mkdir(parents=True, exist_ok=True)
    _persist_encoder_meta(out_dir, model_json, tick_encoder)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_json, f)

    logging.info(
        "Trained %s model - accuracy %.3f profit %.3f", model_type, acc, profit
    )

def train(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str = "logreg",
    optuna_trials: int = 0,
    half_life_days: float = 0.0,
    prune_threshold: float = 0.0,
    calendar_file: Path | None = None,
    symbol_graph: Path | dict | None = None,
    news_sentiment: Path | pd.DataFrame | None = None,
    ensemble: str | None = None,
    neighbor_corr_windows: Iterable[int] | None = None,
    regime_model: Path | dict | None = None,
    per_regime: bool = False,
    filter_noise: bool = False,
    noise_quantile: float = 0.9,
    smote_threshold: float | None = None,
    meta_weights: Sequence[float] | Path | None = None,
    tick_encoder: Path | None = None,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 0.0,
    drift_weight: float = 0.0,
    bayesian_ensembles: int = 0,
    use_meta_label: bool = False,
    take_profit_mult: float = 1.0,
    stop_loss_mult: float = 1.0,
    hold_period: int = 20,
    **kwargs,
) -> None:
    """Public training entry point."""
    graph_path = symbol_graph
    if graph_path is None:
        default_path = data_dir / "symbol_graph.json"
        if default_path.exists():
            graph_path = default_path
        elif Path("symbol_graph.json").exists():
            graph_path = Path("symbol_graph.json")

    ns_df: pd.DataFrame | None = None
    if news_sentiment is not None:
        if isinstance(news_sentiment, pd.DataFrame):
            ns_df = news_sentiment
        else:
            try:
                ns_df = pd.read_csv(news_sentiment)
            except Exception:
                ns_df = None

    if model_type == "transformer":
        return _train_transformer(
            data_dir,
            out_dir,
            calendar_file=calendar_file,
            symbol_graph=graph_path,
            news_sentiment=ns_df,
            neighbor_corr_windows=neighbor_corr_windows,
            tick_encoder=tick_encoder,
            drift_scores=drift_scores,
            drift_threshold=drift_threshold,
            drift_weight=drift_weight,
            bayesian_ensembles=bayesian_ensembles,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            hold_period=hold_period,
            use_meta_label=use_meta_label,
            **kwargs,
        )
    elif model_type == "tabtransformer":
        return _train_tab_transformer(
            data_dir,
            out_dir,
            calendar_file=calendar_file,
            symbol_graph=graph_path,
            news_sentiment=ns_df,
            neighbor_corr_windows=neighbor_corr_windows,
            tick_encoder=tick_encoder,
            drift_scores=drift_scores,
            drift_threshold=drift_threshold,
            drift_weight=drift_weight,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            hold_period=hold_period,
            use_meta_label=use_meta_label,
            epochs=kwargs.get("epochs", 5),
            lr=kwargs.get("lr", 1e-3),
            dropout=kwargs.get("dropout", 0.0),
            device=kwargs.get("device", "cpu"),
        )
    elif model_type in {"xgboost", "lgbm", "catboost"}:
        _train_tree_model(
            data_dir,
            out_dir,
            model_type=model_type,
            half_life_days=half_life_days,
            calendar_file=calendar_file,
            symbol_graph=graph_path,
            news_sentiment=ns_df,
            neighbor_corr_windows=neighbor_corr_windows,
            tick_encoder=tick_encoder,
            drift_scores=drift_scores,
            drift_threshold=drift_threshold,
            drift_weight=drift_weight,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            hold_period=hold_period,
            use_meta_label=use_meta_label,
            **kwargs,
        )
    else:
        _train_lite_mode(
            data_dir,
            out_dir,
            optuna_trials=optuna_trials,
            half_life_days=half_life_days,
            prune_threshold=prune_threshold,
            calendar_file=calendar_file,
            symbol_graph=graph_path,
            news_sentiment=ns_df,
            ensemble=ensemble,
            neighbor_corr_windows=neighbor_corr_windows,
            regime_model=regime_model,
            per_regime=per_regime,
            filter_noise=filter_noise,
            noise_quantile=noise_quantile,
            meta_weights=meta_weights,
            tick_encoder=tick_encoder,
            drift_scores=drift_scores,
            drift_threshold=drift_threshold,
            drift_weight=drift_weight,
            bayesian_ensembles=bayesian_ensembles,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            hold_period=hold_period,
            use_meta_label=use_meta_label,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Resource detection utilities (kept from original implementation)
# ---------------------------------------------------------------------------


def detect_resources():
    """Detect available resources and suggest an operating mode."""
    try:
        mem_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        mem_gb = 0.0
    try:
        swap_gb = psutil.swap_memory().total / (1024**3)
    except Exception:
        swap_gb = 0.0
    try:
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except Exception:
        cores = 0
    try:
        cpu_mhz = psutil.cpu_freq().max
    except Exception:
        cpu_mhz = 0.0
    disk_gb = shutil.disk_usage("/").free / (1024**3)
    lite_mode = mem_gb < 4 or cores < 2 or disk_gb < 5
    heavy_mode = mem_gb >= 8 and cores >= 4

    gpu_mem_gb = 0.0
    has_gpu = False
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                has_gpu = True
        except Exception:
            has_gpu = False
    if not has_gpu:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            gpu_mem_gb = float(out.splitlines()[0])
            has_gpu = True
        except Exception:
            gpu_mem_gb = 0.0
            has_gpu = False

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    CPU_MHZ_THRESHOLD = 2500.0
    if lite_mode:
        model_type = "logreg"
    else:
        model_type = "transformer"
        if not (
            has_gpu
            and has("transformers")
            and gpu_mem_gb >= 8.0
            and cpu_mhz >= CPU_MHZ_THRESHOLD
        ):
            model_type = "logreg"

    use_optuna = heavy_mode and has("optuna")
    bayes_steps = 20 if use_optuna else 0
    enable_rl = (
        heavy_mode and has_gpu and gpu_mem_gb >= 8.0 and has("stable_baselines3")
    )
    if enable_rl:
        mode = "rl"
    elif lite_mode:
        mode = "lite"
    elif model_type != "logreg":
        mode = "deep"
    elif heavy_mode:
        mode = "heavy"
    else:
        mode = "standard"
    return {
        "lite_mode": lite_mode,
        "heavy_mode": heavy_mode,
        "model_type": model_type,
        "bayes_steps": bayes_steps,
        "mem_gb": mem_gb,
        "swap_gb": swap_gb,
        "disk_gb": disk_gb,
        "cores": cores,
        "cpu_mhz": cpu_mhz,
        "has_gpu": has_gpu,
        "gpu_mem_gb": gpu_mem_gb,
        "enable_rl": enable_rl,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Federated helper
# ---------------------------------------------------------------------------


def sync_with_server(
    model_path: Path,
    server_url: str,
    poll_interval: float = 1.0,
    timeout: float = 30.0,
) -> None:
    """Send model weights to a federated server and retrieve aggregated ones."""
    open_func = gzip.open if model_path.suffix == ".gz" else open
    try:
        with open_func(model_path, "rt") as f:
            model = json.load(f)
    except FileNotFoundError:
        return
    weights = model.get("coefficients")
    intercept = model.get("intercept")
    if weights is None:
        return
    payload = {"weights": weights}
    if intercept is not None:
        payload["intercept"] = intercept
    try:
        requests.post(f"{server_url}/update", json=payload, timeout=5)
    except Exception:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/weights", timeout=5)
            data = r.json()
            model["coefficients"] = data.get("weights", model.get("coefficients"))
            if "intercept" in data:
                model["intercept"] = data["intercept"]
            with open_func(model_path, "wt") as f:
                json.dump(model, f)
            break
        except Exception:
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Train target clone model")
    p.add_argument("data_dir", type=Path, help="Directory containing trades_raw.csv")
    p.add_argument("out_dir", type=Path, help="Where to write model.json")
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="minimum accuracy required for at least one fold",
    )
    p.add_argument(
        "--min-profit",
        type=float,
        default=0.0,
        help="minimum profit required for at least one fold",
    )
    p.add_argument("--replay-file", type=Path, help="CSV file with labeled decisions")
    p.add_argument(
        "--replay-weight",
        type=float,
        default=1.0,
        help="sample weight for replay decisions",
    )
    p.add_argument(
        "--uncertain-file",
        type=Path,
        help="CSV with labeled uncertain decisions",
    )
    p.add_argument(
        "--uncertain-weight",
        type=float,
        default=2.0,
        help="sample weight multiplier for uncertain decisions",
    )
    p.add_argument(
        "--half-life-days",
        type=float,
        default=0.0,
        help="half-life in days for exponential sample weight decay",
    )
    p.add_argument(
        "--model-type",
        choices=[
            "logreg",
            "xgboost",
            "lgbm",
            "catboost",
            "transformer",
            "tabtransformer",
        ],
        default="logreg",
        help="which model architecture to train",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device to use for training (e.g. 'cpu' or 'cuda')",
    )
    p.add_argument(
        "--window",
        type=int,
        default=16,
        help="sequence window size for transformer model",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of training epochs for transformer models",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for transformer optimizers",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout rate for transformer layers",
    )
    p.add_argument(
        "--bayesian-ensembles",
        type=int,
        default=0,
        help="number of models for Bayesian ensemble averaging",
    )
    p.add_argument(
        "--ensemble",
        choices=["none", "voting", "stacking"],
        default="none",
        help="optional ensemble method to combine base models",
    )
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="number of Optuna trials for hyperparameter search",
    )
    p.add_argument(
        "--logreg-c-low",
        type=float,
        default=0.01,
        help="lower bound for logistic C in Optuna",
    )
    p.add_argument(
        "--logreg-c-high",
        type=float,
        default=100.0,
        help="upper bound for logistic C in Optuna",
    )
    p.add_argument(
        "--logreg-l1-low",
        type=float,
        default=0.0,
        help="lower bound for elasticnet l1_ratio",
    )
    p.add_argument(
        "--logreg-l1-high",
        type=float,
        default=1.0,
        help="upper bound for elasticnet l1_ratio",
    )
    p.add_argument(
        "--gboost-n-est-low",
        type=int,
        default=50,
        help="lower bound for GradientBoosting n_estimators",
    )
    p.add_argument(
        "--gboost-n-est-high",
        type=int,
        default=200,
        help="upper bound for GradientBoosting n_estimators",
    )
    p.add_argument(
        "--gboost-subsample-low",
        type=float,
        default=0.5,
        help="lower bound for GradientBoosting subsample",
    )
    p.add_argument(
        "--gboost-subsample-high",
        type=float,
        default=1.0,
        help="upper bound for GradientBoosting subsample",
    )
    p.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="drop features with mean |SHAP| below this value",
    )
    p.add_argument("--drift-scores", type=Path, help="JSON file with per-feature drift scores")
    p.add_argument(
        "--drift-threshold",
        type=float,
        default=0.0,
        help="threshold above which features are pruned",
    )
    p.add_argument(
        "--drift-weight",
        type=float,
        default=0.0,
        help="if >0, scale drifted features by this factor instead of dropping",
    )
    p.add_argument("--calendar-file", type=Path, help="CSV file with calendar events")
    p.add_argument(
        "--symbol-graph",
        type=Path,
        help="JSON file with symbol graph (defaults to data_dir/symbol_graph.json)",
    )
    p.add_argument("--news-sentiment", type=Path, help="CSV file with news sentiment")
    p.add_argument(
        "--neighbor-corr-windows",
        type=str,
        help="Comma-separated window sizes for neighbor correlation features",
    )
    p.add_argument(
        "--meta-weights",
        type=Path,
        help="JSON file with meta-learned initial weights",
    )
    p.add_argument(
        "--tick-encoder",
        type=Path,
        help="path to pretrained contrastive encoder",
    )
    p.add_argument(
        "--autoencoder",
        action="store_true",
        help="enable training and application of feature autoencoder",
    )
    p.add_argument(
        "--autoencoder-dim",
        type=int,
        default=8,
        help="dimension of autoencoder bottleneck",
    )
    p.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=10,
        help="number of epochs to train autoencoder",
    )
    p.add_argument(
        "--use-meta-label",
        action="store_true",
        help="use meta labels derived from triple-barrier method",
    )
    p.add_argument(
        "--take-profit-mult",
        type=float,
        default=1.0,
        help="multiplier for take-profit barrier",
    )
    p.add_argument(
        "--stop-loss-mult",
        type=float,
        default=1.0,
        help="multiplier for stop-loss barrier",
    )
    p.add_argument(
        "--hold-period",
        type=int,
        default=20,
        help="maximum holding period for meta labeling",
    )
    args = p.parse_args()
    corr_windows = (
        [int(w) for w in args.neighbor_corr_windows.split(",") if w]
        if args.neighbor_corr_windows
        else None
    )
    drift_scores = None
    if args.drift_scores:
        try:
            drift_scores = json.loads(Path(args.drift_scores).read_text())
        except Exception:
            drift_scores = None
    train(
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        optuna_trials=args.optuna_trials,
        half_life_days=args.half_life_days,
        prune_threshold=args.prune_threshold,
        logreg_c_low=args.logreg_c_low,
        logreg_c_high=args.logreg_c_high,
        logreg_l1_low=args.logreg_l1_low,
        logreg_l1_high=args.logreg_l1_high,
        gboost_n_estimators_low=args.gboost_n_est_low,
        gboost_n_estimators_high=args.gboost_n_est_high,
        gboost_subsample_low=args.gboost_subsample_low,
        gboost_subsample_high=args.gboost_subsample_high,
        min_accuracy=args.min_accuracy,
        min_profit=args.min_profit,
        replay_file=args.replay_file,
        replay_weight=args.replay_weight,
        calendar_file=args.calendar_file,
        symbol_graph=args.symbol_graph,
        news_sentiment=args.news_sentiment,
        ensemble=None if args.ensemble == "none" else args.ensemble,
        neighbor_corr_windows=corr_windows,
        tick_encoder=args.tick_encoder,
        meta_weights=args.meta_weights,
        drift_scores=drift_scores,
        drift_threshold=args.drift_threshold,
        drift_weight=args.drift_weight,
        device=args.device,
        window=args.window,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        bayesian_ensembles=args.bayesian_ensembles,
        use_autoencoder=args.autoencoder,
        autoencoder_dim=args.autoencoder_dim,
        autoencoder_epochs=args.autoencoder_epochs,
        uncertain_file=args.uncertain_file,
        uncertain_weight=args.uncertain_weight,
        use_meta_label=args.use_meta_label,
        take_profit_mult=args.take_profit_mult,
        stop_loss_mult=args.stop_loss_mult,
        hold_period=args.hold_period,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
