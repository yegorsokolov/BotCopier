"""Training pipeline orchestrating model training and evaluation."""
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psutil

try:  # optional polars support
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False
from botcopier.data.loading import _load_logs
from botcopier.features.anomaly import _clip_train_features
from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.features.technical import (
    _extract_features,
    _neutralize_against_market_index,
)
from pydantic import ValidationError

from botcopier.models.registry import MODEL_REGISTRY, get_model
from botcopier.models.schema import ModelParams

try:  # optional torch dependency flag
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # optional mlflow dependency
    import mlflow  # type: ignore

    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False


logger = logging.getLogger(__name__)


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str = "logreg",
    cache_dir: Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    **kwargs: object,
) -> None:
    """Train a model selected from the registry."""
    if cache_dir is not None:
        configure_cache(FeatureConfig(cache_dir=cache_dir))
    df, feature_names, _ = _load_logs(data_dir)
    df, feature_names, _, _ = _extract_features(df, feature_names)
    label_col = next((c for c in df.columns if c.startswith("label")), None)
    if label_col is None:
        raise ValueError("no label column found")
    if isinstance(df, pd.DataFrame):
        y = df[label_col].to_numpy(dtype=float)
        X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
    elif _HAS_POLARS and isinstance(df, pl.DataFrame):
        y = df[label_col].to_numpy().astype(float)
        X = df.select(feature_names).fill_null(0.0).to_numpy().astype(float)
    else:  # pragma: no cover - defensive
        raise TypeError("Unsupported DataFrame type")

    mlflow_active = (tracking_uri is not None) or (experiment_name is not None)
    if mlflow_active and not _HAS_MLFLOW:
        raise RuntimeError("mlflow is required for tracking")
    if mlflow_active:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

    run_ctx = mlflow.start_run() if mlflow_active else nullcontext()
    with run_ctx:
        builder = get_model(model_type)
        model_data, predict_fn = builder(X, y)
        probas = predict_fn(X)
        preds = (probas >= 0.5).astype(float)
        score = float((preds == y).mean())
        model = {"feature_names": feature_names, **model_data}
        if "feature_mean" not in model:
            model["feature_mean"] = X.mean(axis=0).tolist()
        if "feature_std" not in model:
            model["feature_std"] = X.std(axis=0).tolist()
        if "clip_low" not in model:
            model["clip_low"] = np.min(X, axis=0).tolist()
        if "clip_high" not in model:
            model["clip_high"] = np.max(X, axis=0).tolist()
        if "session_models" not in model:
            sm_keys = [
                "coefficients",
                "intercept",
                "feature_mean",
                "feature_std",
                "clip_low",
                "clip_high",
            ]
            model["session_models"] = {
                "asian": {k: model[k] for k in sm_keys if k in model}
            }
        mode = kwargs.get("mode")
        if mode is not None:
            model["mode"] = mode
        out_dir.mkdir(parents=True, exist_ok=True)
        params = ModelParams(**model)
        (out_dir / "model.json").write_text(params.model_dump_json())
        if mlflow_active:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_metric("train_accuracy", float(score))
            mlflow.log_artifact(str(out_dir / "model.json"), artifact_path="model")


def detect_resources(*, lite_mode: bool = False, heavy_mode: bool = False) -> dict:
    """Detect available system resources."""
    vm = psutil.virtual_memory()
    mem = getattr(vm, "available", getattr(vm, "total", 0)) / (1024**3)
    swap = psutil.swap_memory().total / (1024**3)
    disk = shutil.disk_usage("/").free / (1024**3)
    cores = psutil.cpu_count()
    cpu_mhz = getattr(psutil.cpu_freq(), "max", 0.0)
    gpu_mem_gb = 0.0
    has_gpu = False
    if _HAS_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
        has_gpu = True
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
    max_retries: int = 5,
) -> None:
    """Send model weights to a federated server and retrieve aggregated ones.

    Raises
    ------
    RuntimeError
        If communication with ``server_url`` fails after ``max_retries`` attempts.
    """
    open_func = gzip.open if model_path.suffix == ".gz" else open
    try:
        with open_func(model_path, "rt") as f:
            params = ModelParams.model_validate_json(f.read())
    except (FileNotFoundError, ValidationError):
        return

    try:
        import requests
    except Exception as exc:  # pragma: no cover - import failure is rare
        logger.exception("requests dependency not available")
        raise RuntimeError("requests library required to sync with server") from exc

    model = params.model_dump()
    payload = {
        "weights": model.get("coefficients"),
        "intercept": model.get("intercept"),
    }

    delay = poll_interval
    for attempt in range(1, max_retries + 1):
        try:
            requests.post(f"{server_url}/update", json=payload, timeout=5)
            break
        except Exception as exc:
            logger.exception(
                "Failed to post update to %s (attempt %d/%d)",
                server_url,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                raise RuntimeError("Failed to post update to server") from exc
            time.sleep(delay)
            delay *= 2

    deadline = time.time() + timeout
    delay = poll_interval
    attempt = 1
    while time.time() < deadline and attempt <= max_retries:
        try:
            r = requests.get(f"{server_url}/weights", timeout=5)
            data = r.json()
            model["coefficients"] = data.get("weights", model.get("coefficients"))
            if "intercept" in data:
                model["intercept"] = data["intercept"]
            with open_func(model_path, "wt") as f:
                f.write(ModelParams(**model).model_dump_json())
            return
        except Exception as exc:
            logger.exception(
                "Failed to fetch weights from %s (attempt %d/%d)",
                server_url,
                attempt,
                max_retries,
            )
            if attempt == max_retries or time.time() + delay > deadline:
                raise RuntimeError("Failed to retrieve weights from server") from exc
            time.sleep(delay)
            delay *= 2
            attempt += 1

    raise RuntimeError("Timed out retrieving weights from server")


def main() -> None:
    p = argparse.ArgumentParser(description="Train target clone model")
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY.keys()),
        default="logreg",
        help=f"model type to train ({', '.join(MODEL_REGISTRY.keys())})",
    )
    p.add_argument("--tracking-uri", dest="tracking_uri", type=str, default=None)
    p.add_argument("--experiment-name", dest="experiment_name", type=str, default=None)
    args = p.parse_args()
    train(
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
