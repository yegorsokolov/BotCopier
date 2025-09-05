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
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

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
    chunk_size: int = 50000,
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

    optional_cols = ["spread", "slippage", "equity", "margin_level"]
    feature_cols = [c for c in optional_cols if c in df.columns]

    if lite_mode:
        def _iter():
            for start in range(0, len(df), chunk_size):
                yield df.iloc[start:start+chunk_size]
        return _iter(), feature_cols, []
    return df, feature_cols, []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_lite_mode(
    data_dir: Path,
    out_dir: Path,
    *,
    chunk_size: int = 50000,
    hash_size: int = 0,
    flight_uri: str | None = None,
    mode: str = "lite",
    **_: object,
) -> None:
    """Train ``SGDClassifier`` on features from ``trades_raw.csv``."""
    rows_iter, _, _ = _load_logs(
        data_dir, lite_mode=True, chunk_size=chunk_size, flight_uri=flight_uri
    )

    vec = (
        DictVectorizer(sparse=False)
        if hash_size <= 0
        else FeatureHasher(n_features=hash_size, input_type="dict")
    )
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss")
    feature_names: list[str] = []
    first = True
    for chunk in rows_iter:
        if "label" not in chunk.columns:
            continue
        feat_cols = [c for c in chunk.columns if c not in {"label"}]
        feature_names = feat_cols
        f_chunk = chunk[feat_cols].to_dict("records")
        l_chunk = chunk["label"].astype(int).tolist()
        X = (
            vec.fit_transform(f_chunk)
            if first and hash_size <= 0
            else vec.transform(f_chunk)
        )
        if first and hash_size > 0:
            X = vec.transform(f_chunk)
        X = scaler.partial_fit(X).transform(X) if hasattr(scaler, "partial_fit") else scaler.fit_transform(X)
        clf.partial_fit(X, l_chunk, classes=np.array([0, 1]) if first else None)
        first = False
    if first:
        raise ValueError(f"No training data found in {data_dir}")

    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "coefficients": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": 0.5,
        "classes": [int(c) for c in clf.classes_],
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
