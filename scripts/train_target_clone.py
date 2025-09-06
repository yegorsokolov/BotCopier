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
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score

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

    optional_cols = ["spread", "slippage", "equity", "margin_level", "volume"]
    feature_cols = [c for c in optional_cols if c in df.columns]

    # When ``chunk_size`` is provided (or lite_mode explicitly enabled), yield
    # DataFrame chunks instead of a single concatenated frame so callers can
    # control memory usage.  ``lite_mode`` keeps backwards compatibility where
    # chunked iteration was previously the default behaviour.
    cs = chunk_size or (50000 if lite_mode else None)
    if cs:
        def _iter():
            for start in range(0, len(df), cs):
                yield df.iloc[start:start + cs]
        return _iter(), feature_cols, []

    return df, feature_cols, []


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
    **_: object,
) -> None:
    """Train ``SGDClassifier`` on features from ``trades_raw.csv``."""
    df, _, _ = _load_logs(data_dir, chunk_size=chunk_size, flight_uri=flight_uri)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)

    if "label" not in df.columns:
        raise ValueError("label column missing from data")

    feature_names = [c for c in df.columns if c != "label"]

    def _session_from_hour(hour: int) -> str:
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 16:
            return "london"
        return "newyork"

    hours = df["hour"] if "hour" in df.columns else pd.Series([0] * len(df))
    df["session"] = hours.astype(int).apply(_session_from_hour)

    session_models: dict[str, dict[str, object]] = {}
    for name, group in df.groupby("session"):
        if len(group) < 2:
            continue
        feat_cols = [c for c in feature_names if c != "session"]
        X_all = group[feat_cols].to_numpy()
        y_all = group["label"].astype(int).to_numpy()
        split_idx = max(1, int(len(group) * 0.8))
        X_train, X_val = X_all[:split_idx], X_all[split_idx:]
        y_train, y_val = y_all[:split_idx], y_all[split_idx:]
        if len(X_val) == 0:
            continue
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        clf = SGDClassifier(loss="log_loss")
        clf.partial_fit(X_train_scaled, y_train, classes=np.array([0, 1]))
        probs = clf.predict_proba(X_val_scaled)[:, 1]
        thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
        best_thresh = 0.5
        best_acc = -1.0
        best_rec = 0.0
        for t in thresholds:
            preds = (probs >= t).astype(int)
            acc = accuracy_score(y_val, preds)
            rec = recall_score(y_val, preds, zero_division=0)
            if acc > best_acc:
                best_acc = acc
                best_rec = rec
                best_thresh = float(t)
        session_models[name] = {
            "coefficients": clf.coef_[0].astype(float).tolist(),
            "intercept": float(clf.intercept_[0]),
            "threshold": best_thresh,
            "feature_mean": scaler.mean_.astype(float).tolist(),
            "feature_std": scaler.scale_.astype(float).tolist(),
            "metrics": {"accuracy": float(best_acc), "recall": float(best_rec)},
        }

    if not session_models:
        raise ValueError(f"No training data found in {data_dir}")

    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "session_models": session_models,
        "session_hours": {
            "asian": [0, 8],
            "london": [8, 16],
            "newyork": [16, 24],
        },
        "training_mode": "lite",
        "mode": mode,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)


def train(data_dir: Path, out_dir: Path, **kwargs) -> None:
    """Public training entry point."""
    _train_lite_mode(data_dir, out_dir, **kwargs)


# ---------------------------------------------------------------------------
# Resource detection utilities (kept from original implementation)
# ---------------------------------------------------------------------------

def detect_resources():
    """Detect available resources and suggest an operating mode."""
    try:
        mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        mem_gb = 0.0
    try:
        swap_gb = psutil.swap_memory().total / (1024 ** 3)
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
    disk_gb = shutil.disk_usage("/").free / (1024 ** 3)
    lite_mode = mem_gb < 4 or cores < 2 or disk_gb < 5
    heavy_mode = mem_gb >= 8 and cores >= 4

    gpu_mem_gb = 0.0
    has_gpu = False
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                gpu_mem_gb = (
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
            has_gpu and has("transformers") and gpu_mem_gb >= 8.0 and cpu_mhz >= CPU_MHZ_THRESHOLD
        ):
            model_type = "logreg"

    use_optuna = heavy_mode and has("optuna")
    bayes_steps = 20 if use_optuna else 0
    enable_rl = heavy_mode and has_gpu and gpu_mem_gb >= 8.0 and has("stable_baselines3")
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
    args = p.parse_args()
    train(args.data_dir, args.out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
