"""Training pipeline orchestrating model training and evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features, configure_cache

try:  # optional torch dependency flag
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str = "logreg",
    cache_dir: Path | None = None,
    **_: object,
) -> None:
    """Train a simple logistic regression model from trade logs."""
    if cache_dir is not None:
        configure_cache(cache_dir)
    df, feature_names, _ = _load_logs(data_dir)
    df, feature_names, _, _ = _extract_features(df, feature_names)
    label_col = next((c for c in df.columns if c.startswith("label")), None)
    if label_col is None:
        raise ValueError("no label column found")
    y = df[label_col].to_numpy(dtype=float)
    X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    model = {
        "feature_names": feature_names,
        "coefficients": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "feature_mean": X.mean(axis=0).tolist(),
        "feature_std": X.std(axis=0).tolist(),
        "clip_low": np.min(X, axis=0).tolist(),
        "clip_high": np.max(X, axis=0).tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)


def detect_resources(*, lite_mode: bool = False, heavy_mode: bool = False) -> dict:
    """Detect available system resources."""
    vm = psutil.virtual_memory()
    mem = getattr(vm, "available", getattr(vm, "total", 0)) / (1024 ** 3)
    swap = psutil.swap_memory().total / (1024 ** 3)
    disk = shutil.disk_usage("/").free / (1024 ** 3)
    cores = psutil.cpu_count()
    cpu_mhz = getattr(psutil.cpu_freq(), "max", 0.0)
    gpu_mem_gb = 0.0
    has_gpu = False
    if _HAS_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
        has_gpu = True
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    model_type = "logreg"
    if has_gpu and gpu_mem_gb >= 8.0:
        model_type = "transformer"
    CPU_MHZ_THRESHOLD = 2500.0
    heavy_mode = heavy_mode or cpu_mhz >= CPU_MHZ_THRESHOLD
    enable_rl = has_gpu and gpu_mem_gb >= 8.0
    mode = "standard"
    if enable_rl:
        mode = "rl"
    elif lite_mode:
        mode = "lite"
    elif heavy_mode:
        mode = "heavy"
    return {
        "lite_mode": lite_mode,
        "heavy_mode": heavy_mode,
        "model_type": model_type,
        "mem_gb": mem,
        "swap_gb": swap,
        "disk_gb": disk,
        "cores": cores,
        "gpu_mem_gb": gpu_mem_gb,
        "has_gpu": has_gpu,
        "mode": mode,
        "cpu_mhz": cpu_mhz,
    }


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
    try:
        import requests

        payload = {
            "weights": model.get("coefficients"),
            "intercept": model.get("intercept"),
        }
        requests.post(f"{server_url}/update", json=payload, timeout=5)
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{server_url}/weights", timeout=5)
            data = r.json()
            model["coefficients"] = data.get("weights", model.get("coefficients"))
            if "intercept" in data:
                model["intercept"] = data["intercept"]
            with open_func(model_path, "wt") as f:
                json.dump(model, f)
            break
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Train target clone model")
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()
    train(args.data_dir, args.out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
