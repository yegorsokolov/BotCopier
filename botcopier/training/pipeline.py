"""Training pipeline orchestrating model training and evaluation."""
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil

try:  # optional polars support
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False
from pydantic import ValidationError

from botcopier.data.loading import _load_logs
from botcopier.features.anomaly import _clip_train_features
from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.features.technical import (
    _extract_features,
    _neutralize_against_market_index,
)
from botcopier.models.registry import MODEL_REGISTRY, get_model, load_params
from botcopier.models.schema import ModelParams
from botcopier.scripts.evaluation import _classification_metrics
from botcopier.scripts.splitters import PurgedWalkForward
from logging_utils import setup_logging

try:  # optional feast dependency
    from feast import FeatureStore  # type: ignore

    from botcopier.feature_store.feast_repo.feature_views import FEATURE_COLUMNS

    _HAS_FEAST = True
except Exception:  # pragma: no cover - optional
    FeatureStore = None  # type: ignore
    FEATURE_COLUMNS = []  # type: ignore
    _HAS_FEAST = False

try:  # optional torch dependency flag
    import torch  # type: ignore

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # optional mlflow dependency
    import mlflow  # type: ignore

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

try:  # optional ray dependency
    import ray  # type: ignore

    _HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore
    _HAS_RAY = False


logger = logging.getLogger(__name__)


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str = "logreg",
    cache_dir: Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    features: Sequence[str] | None = None,
    distributed: bool = False,
    **kwargs: object,
) -> None:
    """Train a model selected from the registry."""
    configure_cache(
        FeatureConfig(cache_dir=cache_dir, enabled_features=set(features or []))
    )
    load_keys = [
        "lite_mode",
        "chunk_size",
        "flight_uri",
        "kafka_brokers",
        "take_profit_mult",
        "stop_loss_mult",
        "hold_period",
        "augment_ratio",
        "dtw_augment",
    ]
    load_kwargs = {k: kwargs[k] for k in load_keys if k in kwargs}
    logs, feature_names, _ = _load_logs(data_dir, **load_kwargs)
    fs_repo = Path(__file__).resolve().parents[1] / "feature_store" / "feast_repo"
    y_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    profit_list: list[np.ndarray] = []
    label_col: str | None = None
    if (
        isinstance(logs, Iterable)
        and not isinstance(logs, (pd.DataFrame,))
        and not (_HAS_POLARS and isinstance(logs, pl.DataFrame))
    ):
        store = FeatureStore(repo_path=str(fs_repo)) if _HAS_FEAST else None
        feature_refs = [f"trade_features:{f}" for f in FEATURE_COLUMNS]
        for chunk in logs:
            if _HAS_FEAST and store is not None:
                feat_df = store.get_historical_features(
                    entity_df=chunk, features=feature_refs
                ).to_df()
                chunk = chunk.merge(feat_df, on=["symbol", "event_time"], how="left")
                feature_names = list(FEATURE_COLUMNS)
            else:
                chunk, feature_names, _, _ = _extract_features(chunk, feature_names)
            if label_col is None:
                label_col = next(
                    (c for c in chunk.columns if c.startswith("label")), None
                )
                if label_col is None:
                    raise ValueError("no label column found")
            y_list.append(chunk[label_col].to_numpy(dtype=float))
            X_list.append(chunk[feature_names].fillna(0.0).to_numpy(dtype=float))
            if "profit" in chunk.columns:
                profit_list.append(chunk["profit"].to_numpy(dtype=float))
        y = np.concatenate(y_list, axis=0)
        X = np.vstack(X_list)
        profits = (
            np.concatenate(profit_list, axis=0) if profit_list else np.zeros_like(y)
        )
    else:
        df = logs  # type: ignore[assignment]
        if _HAS_FEAST:
            store = FeatureStore(repo_path=str(fs_repo))
            feature_refs = [f"trade_features:{f}" for f in FEATURE_COLUMNS]
            feat_df = store.get_historical_features(
                entity_df=df, features=feature_refs
            ).to_df()
            df = df.merge(feat_df, on=["symbol", "event_time"], how="left")
            feature_names = list(FEATURE_COLUMNS)
        else:
            df, feature_names, _, _ = _extract_features(df, feature_names)
        label_col = next((c for c in df.columns if c.startswith("label")), None)
        if label_col is None:
            raise ValueError("no label column found")
        if isinstance(df, pd.DataFrame):
            y = df[label_col].to_numpy(dtype=float)
            X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
            profits = (
                df["profit"].to_numpy(dtype=float)
                if "profit" in df.columns
                else np.zeros_like(y)
            )
        elif _HAS_POLARS and isinstance(df, pl.DataFrame):
            y = df[label_col].to_numpy().astype(float)
            X = df.select(feature_names).fill_null(0.0).to_numpy().astype(float)
            profits = (
                df["profit"].to_numpy().astype(float)
                if "profit" in df.columns
                else np.zeros_like(y)
            )
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
        n_splits = int(kwargs.get("n_splits", 3))
        gap = int(kwargs.get("cv_gap", 1))
        param_grid = kwargs.get("param_grid") or [{}]
        best_score = -np.inf
        best_params: dict[str, object] | None = None
        best_fold_metrics: list[dict[str, object]] = []
        best_agg: dict[str, float] = {}
        if distributed and not _HAS_RAY:
            raise RuntimeError("ray is required for distributed execution")
        for params in param_grid:
            splitter = PurgedWalkForward(n_splits=n_splits, gap=gap)
            fold_metrics: list[dict[str, object]] = []
            if distributed and _HAS_RAY:
                X_ref = ray.put(X)
                y_ref = ray.put(y)
                profits_ref = ray.put(profits)

                @ray.remote
                def _run_fold(tr_idx, val_idx, fold):
                    X = ray.get(X_ref)
                    y = ray.get(y_ref)
                    profits = ray.get(profits_ref)
                    builder = get_model(model_type)
                    model_fold, pred_fn = builder(X[tr_idx], y[tr_idx], **params)
                    prob_val = pred_fn(X[val_idx])
                    returns = profits[val_idx] * (prob_val >= 0.5)
                    metrics = _classification_metrics(y[val_idx], prob_val, returns)
                    return fold, metrics

                futures = [
                    _run_fold.remote(tr_idx, val_idx, fold)
                    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X))
                    if len(np.unique(y[tr_idx])) >= 2
                ]
                results = ray.get(futures)
                results.sort(key=lambda x: x[0])
                for fold, metrics in results:
                    fold_metrics.append(metrics)
                    logger.info(
                        "Fold %d params %s metrics %s", fold + 1, params, metrics
                    )
                    if mlflow_active:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                mlflow.log_metric(f"fold{fold + 1}_{k}", float(v))
            else:
                for fold, (tr_idx, val_idx) in enumerate(splitter.split(X)):
                    if len(np.unique(y[tr_idx])) < 2:
                        continue
                    builder = get_model(model_type)
                    model_fold, pred_fn = builder(X[tr_idx], y[tr_idx], **params)
                    prob_val = pred_fn(X[val_idx])
                    returns = profits[val_idx] * (prob_val >= 0.5)
                    metrics = _classification_metrics(y[val_idx], prob_val, returns)
                    fold_metrics.append(metrics)
                    logger.info(
                        "Fold %d params %s metrics %s", fold + 1, params, metrics
                    )
                    if mlflow_active:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                mlflow.log_metric(f"fold{fold + 1}_{k}", float(v))
            if not fold_metrics:
                continue
            agg = {
                k: float(
                    np.nanmean(
                        [
                            m[k]
                            for m in fold_metrics
                            if isinstance(m.get(k), (int, float))
                        ]
                    )
                )
                for k in fold_metrics[0].keys()
                if k != "reliability_curve"
            }
            score = agg.get("roc_auc", float("nan"))
            if np.isnan(score):
                score = agg.get("accuracy", 0.0)
            logger.info("Aggregated metrics for params %s: %s", params, agg)
            if score > best_score:
                best_score = score
                best_params = params
                best_fold_metrics = fold_metrics
                best_agg = agg
        min_acc = float(kwargs.get("min_accuracy", 0.0))
        min_profit = float(kwargs.get("min_profit", -np.inf))
        if best_agg and (
            best_agg.get("accuracy", 0.0) < min_acc
            or best_agg.get("profit", 0.0) < min_profit
        ):
            raise ValueError("Cross-validation metrics below thresholds")
        builder = get_model(model_type)
        model_data, predict_fn = builder(X, y, **(best_params or {}))

        # --- SHAP based feature selection ---------------------------------
        shap_threshold = float(kwargs.get("shap_threshold", 0.0))
        try:
            import shap  # type: ignore

            explainer = None
            if model_type == "logreg":

                class _LRWrap:
                    def __init__(self, coef: list[float], intercept: float) -> None:
                        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)
                        self.intercept_ = np.asarray([intercept], dtype=float)

                explainer = shap.LinearExplainer(
                    _LRWrap(
                        model_data.get("coefficients", []),
                        float(model_data.get("intercept", 0.0)),
                    ),
                    X,
                )
            else:
                # fall back to TreeExplainer if model exposes tree structure
                if hasattr(model_data, "booster") or model_type in {
                    "xgboost",
                    "lightgbm",
                    "random_forest",
                }:
                    try:
                        explainer = shap.TreeExplainer(predict_fn)  # type: ignore[arg-type]
                    except Exception:  # pragma: no cover - optional
                        explainer = None

            if explainer is not None:
                shap_values = explainer.shap_values(X)
                mean_abs = np.abs(shap_values).mean(axis=0)
                ranking = sorted(
                    zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True
                )
                logger.info("SHAP importance ranking: %s", ranking)
                if shap_threshold > 0.0:
                    mask = mean_abs >= shap_threshold
                    if mask.sum() < len(feature_names):
                        X = X[:, mask]
                        feature_names = [
                            fn for fn, keep in zip(feature_names, mask) if keep
                        ]
                        model_data, predict_fn = builder(X, y, **(best_params or {}))
        except Exception:  # pragma: no cover - shap is optional
            logger.exception("Failed to compute SHAP values")

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
        model["cv_accuracy"] = best_agg.get("accuracy", 0.0)
        model["cv_profit"] = best_agg.get("profit", 0.0)
        model["conformal_lower"] = 0.0
        model["conformal_upper"] = 1.0
        model["session_models"]["asian"]["cv_metrics"] = best_fold_metrics
        model["session_models"]["asian"]["conformal_lower"] = 0.0
        model["session_models"]["asian"]["conformal_upper"] = 1.0
        mode = kwargs.get("mode")
        if mode is not None:
            model["mode"] = mode
        out_dir.mkdir(parents=True, exist_ok=True)
        params = ModelParams(**model)
        (out_dir / "model.json").write_text(params.model_dump_json())
        model_obj = getattr(predict_fn, "model", None)
        if model_obj is not None:
            try:
                from botcopier.onnx_utils import export_model

                export_model(model_obj, X, out_dir / "model.onnx")
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to export ONNX model")
        if mlflow_active:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_metric("train_accuracy", float(score))
            for k, v in best_agg.items():
                mlflow.log_metric(f"cv_{k}", float(v))
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
    try:
        params = load_params(model_path)
    except (FileNotFoundError, ValidationError):
        return
    open_func = gzip.open if model_path.suffix == ".gz" else open

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - import failure is rare
        logger.exception("requests dependency not available")
        raise RuntimeError("requests library required to sync with server") from exc

    model = params.model_dump()
    payload = {
        "weights": model.get("coefficients"),
        "intercept": model.get("intercept"),
    }

    with requests.Session() as session:
        delay = poll_interval
        for attempt in range(1, max_retries + 1):
            try:
                session.post(f"{server_url}/update", json=payload, timeout=5)
                break
            except requests.RequestException as exc:
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
                r = session.get(f"{server_url}/weights", timeout=5)
                data = r.json()
                model["coefficients"] = data.get("weights", model.get("coefficients"))
                if "intercept" in data:
                    model["intercept"] = data["intercept"]
                with open_func(model_path, "wt") as f:
                    f.write(ModelParams(**model).model_dump_json())
                return
            except requests.RequestException as exc:
                logger.exception(
                    "Failed to fetch weights from %s (attempt %d/%d)",
                    server_url,
                    attempt,
                    max_retries,
                )
                if attempt == max_retries or time.time() + delay > deadline:
                    raise RuntimeError(
                        "Failed to retrieve weights from server"
                    ) from exc
                time.sleep(delay)
                delay *= 2
                attempt += 1

    raise RuntimeError("Timed out retrieving weights from server")


def main() -> None:
    setup_logging()
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
