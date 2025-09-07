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
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple, Callable

import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score
from .splitters import PurgedWalkForward
from sklearn.calibration import CalibratedClassifierCV

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
) -> tuple[pd.DataFrame, list[str], dict[str, list[float]]]:
    """Attach graph embeddings and calendar flags."""

    embeddings: dict[str, list[float]] = {}
    if symbol_graph is not None:
        if not isinstance(symbol_graph, dict):
            with open(symbol_graph) as f_sg:
                graph_data = json.load(f_sg)
        else:
            graph_data = symbol_graph
        nodes = graph_data.get("nodes") or {}
        if nodes:
            for sym, vals in nodes.items():
                emb = vals.get("embedding")
                if isinstance(emb, list):
                    embeddings[sym] = [float(v) for v in emb]
        elif graph_data.get("embeddings"):
            embeddings = {
                sym: [float(v) for v in emb]
                for sym, emb in graph_data.get("embeddings", {}).items()
            }
        if embeddings and "symbol" in df.columns:
            emb_dim = len(next(iter(embeddings.values())))
            sym_series = df["symbol"]
            for i in range(emb_dim):
                col = f"graph_emb{i}"
                df[col] = sym_series.map(
                    lambda s: (
                        embeddings.get(str(s), [0.0] * emb_dim)[i]
                        if isinstance(s, str)
                        else 0.0
                    )
                )
            feature_names = feature_names + [f"graph_emb{i}" for i in range(emb_dim)]

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
            direction="backward",
        )
        merged["news_sentiment"] = merged["score"].fillna(0.0)
        merged.drop(columns=["timestamp", "score"], inplace=True)
        df = merged.set_index(df_idx).sort_index()
        feature_names = feature_names + ["news_sentiment"]
    return df, feature_names, embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    symbol_graph: dict | str | Path | None = None,
    calendar_file: Path | None = None,
    optuna_trials: int = 0,
    half_life_days: float = 0.0,
    prune_threshold: float = 0.0,
    purge_gap: int = 1,
    news_sentiment: pd.DataFrame | None = None,
    **_: object,
) -> None:
    """Train ``SGDClassifier`` on features from ``trades_raw.csv``."""
    df, feature_names, _ = _load_logs(
        data_dir, chunk_size=chunk_size, flight_uri=flight_uri
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    feature_names = list(feature_names)
    if replay_file:
        rdf = pd.read_csv(replay_file)
        rdf.columns = [c.lower() for c in rdf.columns]
        df = pd.concat([df, rdf], ignore_index=True)
    if "net_profit" in df.columns:
        weights = pd.to_numeric(df["net_profit"], errors="coerce").abs().to_numpy()
    elif "lots" in df.columns:
        weights = pd.to_numeric(df["lots"], errors="coerce").abs().to_numpy()
    else:
        weights = np.ones(len(df), dtype=float)
    if "lots" in df.columns:
        lot_vals = pd.to_numeric(df["lots"], errors="coerce").abs().to_numpy()
        weights = np.where(weights > 0, weights, lot_vals)
    if replay_file:
        weights[-len(rdf) :] *= replay_weight

    # Compute sample age relative to the most recent trade
    if "event_time" in df.columns:
        ref_time = df["event_time"].max()
        age_days = (ref_time - df["event_time"]).dt.total_seconds() / (24 * 3600)
    else:
        age_days = (df.index.max() - df.index).astype(float)
    df["age_days"] = age_days

    if half_life_days > 0:
        decay = 0.5 ** (age_days / half_life_days)
        weights = weights * decay

    df["sample_weight"] = weights

    if "label" not in df.columns:
        raise ValueError("label column missing from data")

    df, feature_names, embeddings = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
    )
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
    optuna_info: dict[str, object] | None = None
    if optuna_trials > 0 and _HAS_OPTUNA:
        X_all = df[feature_names].to_numpy()
        y_all = df["label"].astype(int).to_numpy()
        sw_all = df["sample_weight"].to_numpy()

        def objective(trial: "optuna.Trial") -> float:
            model_type = trial.suggest_categorical("model_type", ["sgd", "gboost"])
            threshold = trial.suggest_float("threshold", 0.1, 0.9)
            if model_type == "sgd":
                lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            else:
                lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
            tscv = PurgedWalkForward(n_splits=min(3, len(X_all) - 1), gap=purge_gap)
            scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_all):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]
                w_train = sw_all[train_idx]
                if model_type == "sgd":
                    scaler = StandardScaler().fit(X_train)
                    clf = SGDClassifier(
                        loss="log_loss",
                        learning_rate="constant",
                        eta0=lr,
                    )
                    clf.partial_fit(
                        scaler.transform(X_train),
                        y_train,
                        classes=np.array([0, 1]),
                        sample_weight=w_train,
                    )
                    probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
                else:
                    clf = GradientBoostingClassifier(learning_rate=lr, max_depth=depth)
                    clf.fit(X_train, y_train, sample_weight=w_train)
                    probs = clf.predict_proba(X_val)[:, 1]
                preds = (probs >= threshold).astype(int)
                scores.append(accuracy_score(y_val, preds))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=optuna_trials)
        optuna_info = {
            "params": study.best_params,
            "score": float(study.best_value),
        }

    def _session_from_hour(hour: int) -> str:
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 16:
            return "london"
        return "newyork"

    hours = df["hour"] if "hour" in df.columns else pd.Series([0] * len(df))
    df["session"] = hours.astype(int).apply(_session_from_hour)

    session_models: dict[str, dict[str, object]] = {}
    cv_acc_all: list[float] = []
    cv_profit_all: list[float] = []
    for name, group in df.groupby("session"):
        if len(group) < 2:
            continue
        feat_cols = [c for c in feature_names if c != "session"]
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
        fold_metrics: list[dict[str, float]] = []
        fold_thresholds: list[float] = []
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
            scaler = StandardScaler().fit(X_train)
            clf = SGDClassifier(loss="log_loss")
            clf.partial_fit(
                scaler.transform(X_train),
                y_train,
                classes=np.array([0, 1]),
                sample_weight=w_train,
            )
            probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
            # collect probabilities and labels for conformal interval computation
            all_probs.append(probs)
            all_labels.append(y_val)
            thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
            best_thresh = 0.5
            best_acc = -1.0
            best_rec = 0.0
            best_profit = 0.0
            profits_val = (
                group.iloc[val_idx][profit_col].to_numpy()
                if profit_col
                else np.zeros_like(y_val, dtype=float)
            )
            for t in thresholds:
                preds = (probs >= t).astype(int)
                acc = accuracy_score(y_val, preds)
                rec = recall_score(y_val, preds, zero_division=0)
                profit = (
                    float((profits_val * preds).mean()) if len(profits_val) else 0.0
                )
                if acc > best_acc:
                    best_acc = float(acc)
                    best_rec = float(rec)
                    best_thresh = float(t)
                    best_profit = profit
            fold_metrics.append(
                {"accuracy": best_acc, "recall": best_rec, "profit": best_profit}
            )
            fold_thresholds.append(best_thresh)
        if not any(
            fm["accuracy"] >= min_accuracy or fm["profit"] >= min_profit
            for fm in fold_metrics
        ):
            raise ValueError(
                f"Session {name} failed to meet min accuracy {min_accuracy} or profit {min_profit}"
            )
        mean_acc = float(np.mean([fm["accuracy"] for fm in fold_metrics]))
        mean_rec = float(np.mean([fm["recall"] for fm in fold_metrics]))
        mean_profit = float(np.mean([fm["profit"] for fm in fold_metrics]))
        avg_thresh = float(np.mean(fold_thresholds))
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
        scaler_full = StandardScaler().fit(X_all)
        clf_full = SGDClassifier(loss="log_loss")
        X_scaled_full = scaler_full.transform(X_all)
        clf_full.partial_fit(
            X_scaled_full,
            y_all,
            classes=np.array([0, 1]),
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
            "feature_mean": scaler_full.mean_.astype(float).tolist(),
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
        if lot_model:
            params["lot_model"] = lot_model
        if sl_model:
            params["sl_model"] = sl_model
        if tp_model:
            params["tp_model"] = tp_model
        session_models[name] = params
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
            scaler_sym = StandardScaler().fit(X_sym)
            base_clf = SGDClassifier(loss="log_loss")
            base_clf.partial_fit(
                scaler_sym.transform(X_sym),
                y_sym,
                classes=np.array([0, 1]),
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
    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "half_life_days": float(half_life_days),
        "session_models": session_models,
        "session_hours": {
            "asian": [0, 8],
            "london": [8, 16],
            "newyork": [16, 24],
        },
        "training_mode": "lite",
        "mode": mode,
        "cv_accuracy": float(np.mean(cv_acc_all)) if cv_acc_all else 0.0,
        "cv_profit": float(np.mean(cv_profit_all)) if cv_profit_all else 0.0,
        "conformal_lower": float(min(conf_lowers)) if conf_lowers else 0.0,
        "conformal_upper": float(max(conf_uppers)) if conf_uppers else 1.0,
    }
    if symbol_thresholds:
        model["symbol_thresholds"] = symbol_thresholds

    # ------------------------------------------------------------------
    # Train alternative base models and a simple gating network
    # ------------------------------------------------------------------
    def _fit_base_model(X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Fit logistic regression and return params and predictor."""
        scaler = StandardScaler().fit(X)
        clf = SGDClassifier(loss="log_loss")
        clf.partial_fit(
            scaler.transform(X),
            y,
            classes=np.array([0, 1]),
            sample_weight=w,
        )

        def _predict(inp: np.ndarray) -> np.ndarray:
            return clf.predict_proba(scaler.transform(inp))[:, 1]

        params = {
            "coefficients": clf.coef_[0].astype(float).tolist(),
            "intercept": float(clf.intercept_[0]),
            "threshold": 0.5,
            "feature_mean": scaler.mean_.astype(float).tolist(),
            "feature_std": scaler.scale_.astype(float).tolist(),
            "conformal_lower": 0.0,
            "conformal_upper": 1.0,
        }
        return params, _predict

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

    # Fit once to compute SHAP importances and optionally prune features
    params_generic, _ = _fit_base_model(X_all, y_all, w_all)
    coeffs = np.array(params_generic["coefficients"])
    mean = np.array(params_generic["feature_mean"])
    std = np.array(params_generic["feature_std"])
    shap_vals = ((X_all - mean) / std) * coeffs
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    keep_mask = mean_abs_shap >= prune_threshold
    if prune_threshold > 0.0 and not keep_mask.all():
        feat_cols = [n for n, k in zip(feat_cols, keep_mask) if k]
        X_all = X_all[:, keep_mask]
        # Recompute model on reduced feature set
        params_generic, _ = _fit_base_model(X_all, y_all, w_all)

    feature_names = feat_cols

    models: dict[str, dict[str, object]] = {}
    pred_funcs: list[Callable[[np.ndarray], np.ndarray]] = []

    params_generic, pred_generic = _fit_base_model(X_all, y_all, w_all)
    models["logreg"] = params_generic
    pred_funcs.append(pred_generic)

    # Compute feature importance for the final model
    coeffs = np.array(params_generic["coefficients"])
    mean = np.array(params_generic["feature_mean"])
    std = np.array(params_generic["feature_std"])
    shap_vals = ((X_all - mean) / std) * coeffs
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    feature_importance = {n: float(v) for n, v in zip(feature_names, mean_abs_shap)}

    if high_mask.sum() >= 2:
        params_high, pred_high = _fit_base_model(
            X_all[high_mask], y_all[high_mask], w_all[high_mask]
        )
    else:
        params_high, pred_high = params_generic, pred_generic
    models["xgboost"] = params_high
    pred_funcs.append(pred_high)

    if low_mask.sum() >= 2:
        params_low, pred_low = _fit_base_model(
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
    r_mean = router_feats.mean(axis=0)
    r_std = router_feats.std(axis=0)
    r_std[r_std == 0] = 1.0
    norm_router = (router_feats - r_mean) / r_std
    router_clf = SGDClassifier(loss="log_loss")
    router_clf.partial_fit(
        norm_router, best_idx, classes=np.array([0, 1, 2]), sample_weight=w_all
    )
    router = {
        "intercept": router_clf.intercept_.astype(float).tolist(),
        "coefficients": router_clf.coef_.astype(float).tolist(),
        "feature_mean": r_mean.astype(float).tolist(),
        "feature_std": r_std.astype(float).tolist(),
    }
    model["ensemble_router"] = router
    if embeddings:
        model["symbol_embeddings"] = embeddings
    if optuna_info:
        model["optuna_best_params"] = optuna_info.get("params", {})
        model["optuna_best_score"] = optuna_info.get("score", 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)


def _train_transformer(
    data_dir: Path,
    out_dir: Path,
    *,
    window: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    calendar_file: Path | None = None,
    symbol_graph: dict | str | Path | None = None,
    news_sentiment: pd.DataFrame | None = None,
    synthetic_model: Path | None = None,
    synthetic_frac: float = 0.0,
    synthetic_weight: float = 0.2,
) -> None:
    """Train a tiny attention encoder on rolling feature windows."""
    if not _HAS_TORCH:  # pragma: no cover - requires optional dependency
        raise ImportError("PyTorch is required for transformer model")

    df, feature_names, _ = _load_logs(data_dir)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)
    if "label" not in df.columns:
        raise ValueError("label column missing from data")

    df, feature_names, _ = _extract_features(
        df,
        feature_names,
        symbol_graph=symbol_graph,
        calendar_file=calendar_file,
        news_sentiment=news_sentiment,
    )

    X_all = df[feature_names].to_numpy(dtype=float)
    y_all = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    feat_mean = X_all.mean(axis=0)
    feat_std = X_all.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    norm_X = (X_all - feat_mean) / feat_std

    seqs: list[list[list[float]]] = []
    ys: list[float] = []
    synth_flags: list[float] = []
    for i in range(window, len(norm_X)):
        seqs.append(norm_X[i - window : i].tolist())
        ys.append(y_all[i])
        synth_flags.append(0.0)

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
        except Exception:
            pass

    weights_arr = np.where(np.array(synth_flags) > 0.0, synthetic_weight, 1.0).astype(
        float
    )
    X = torch.tensor(seqs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    w = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(-1)
    flags = torch.tensor(synth_flags, dtype=torch.float32).unsqueeze(-1)
    ds = torch.utils.data.TensorDataset(X, y, w, flags)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    class TinyTransformer(torch.nn.Module):
        def __init__(self, in_dim: int, dim: int = 16):
            super().__init__()
            self.embed = torch.nn.Linear(in_dim, dim)
            self.q = torch.nn.Linear(dim, dim)
            self.k = torch.nn.Linear(dim, dim)
            self.v = torch.nn.Linear(dim, dim)
            self.out = torch.nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F)
            emb = self.embed(x)
            q = self.q(emb)
            k = self.k(emb)
            v = self.v(emb)
            attn = torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1
            )
            ctx = torch.matmul(attn, v).mean(dim=1)
            return self.out(ctx)

    dim = 16
    model = TinyTransformer(len(feature_names), dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for _ in range(epochs):  # pragma: no cover - simple training loop
        for batch_x, batch_y, batch_w, _ in dl:
            opt.zero_grad()
            logits = model(batch_x)
            loss = (loss_fn(logits, batch_y) * batch_w).mean()
            loss.backward()
            opt.step()

    def _tensor_list(t: torch.Tensor) -> list:
        return t.detach().cpu().numpy().tolist()

    weights = {
        "embed_weight": _tensor_list(model.embed.weight),
        "embed_bias": _tensor_list(model.embed.bias),
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
        teacher_logits = model(X).squeeze(-1)
        teacher_probs = torch.sigmoid(teacher_logits).cpu().numpy()
    pred_labels = (teacher_probs > 0.5).astype(int)
    y_arr = np.array(ys)
    flags_arr = np.array(synth_flags, dtype=bool)
    metrics_all = {
        "accuracy": float(accuracy_score(y_arr, pred_labels)),
        "recall": float(recall_score(y_arr, pred_labels, zero_division=0)),
    }
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
        "teacher_metrics": teacher_metrics,
        "synthetic_metrics": synthetic_info,
        "distilled": distilled,
        "models": {"logreg": distilled},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_json, f)


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
        _train_transformer(
            data_dir,
            out_dir,
            calendar_file=calendar_file,
            symbol_graph=graph_path,
            news_sentiment=ns_df,
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
        "--half-life-days",
        type=float,
        default=0.0,
        help="half-life in days for exponential sample weight decay",
    )
    p.add_argument(
        "--model-type",
        choices=["logreg", "transformer"],
        default="logreg",
        help="which model architecture to train",
    )
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="number of Optuna trials for hyperparameter search",
    )
    p.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="drop features with mean |SHAP| below this value",
    )
    p.add_argument("--calendar-file", type=Path, help="CSV file with calendar events")
    p.add_argument(
        "--symbol-graph",
        type=Path,
        help="JSON file with symbol graph (defaults to data_dir/symbol_graph.json)",
    )
    p.add_argument("--news-sentiment", type=Path, help="CSV file with news sentiment")
    args = p.parse_args()
    train(
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        optuna_trials=args.optuna_trials,
        half_life_days=args.half_life_days,
        prune_threshold=args.prune_threshold,
        min_accuracy=args.min_accuracy,
        min_profit=args.min_profit,
        replay_file=args.replay_file,
        replay_weight=args.replay_weight,
        calendar_file=args.calendar_file,
        symbol_graph=args.symbol_graph,
        news_sentiment=args.news_sentiment,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
