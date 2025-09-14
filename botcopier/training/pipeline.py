"""Training pipeline orchestrating model training and evaluation."""

from __future__ import annotations

import argparse
import cProfile
import gzip
import importlib.metadata as importlib_metadata
import json
import logging
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler

try:  # optional polars support
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - optional
    pl = None  # type: ignore
    _HAS_POLARS = False

try:  # optional dask support
    import dask.dataframe as dd  # type: ignore

    _HAS_DASK = True
except Exception:  # pragma: no cover - optional
    dd = None  # type: ignore
    _HAS_DASK = False
from opentelemetry import trace
from pydantic import ValidationError

import botcopier.features.technical as technical_features
from automl.controller import AutoMLController
from botcopier.config.settings import TrainingConfig
from botcopier.data.feature_schema import FeatureSchema
from botcopier.data.loading import _load_logs
from botcopier.features.anomaly import _clip_train_features
from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.features.technical import (
    _extract_features,
    _neutralize_against_market_index,
)
from botcopier.models.registry import MODEL_REGISTRY, get_model, load_params
from botcopier.models.schema import FeatureMetadata, ModelParams
from botcopier.scripts.evaluation import _classification_metrics
from botcopier.scripts.model_card import generate_model_card
from botcopier.scripts.portfolio import hierarchical_risk_parity
from botcopier.scripts.splitters import PurgedWalkForward
from botcopier.training.curriculum import _apply_curriculum
from botcopier.utils.random import set_seed
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


def _max_drawdown(returns: np.ndarray) -> float:
    """Return the maximum drawdown of ``returns``."""
    if returns.size == 0:
        return 0.0
    cum = np.cumsum(returns, dtype=float)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd))


def _var_95(returns: np.ndarray) -> float:
    """Return the 95% Value at Risk of ``returns``."""
    if returns.size == 0:
        return 0.0
    return float(-np.quantile(returns, 0.05))


def _write_dependency_snapshot(out_dir: Path) -> Path:
    """Record the current Python package versions."""
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib_metadata.distributions()
    )
    dep_path = out_dir / "dependencies.txt"
    dep_path.write_text("\n".join(packages))
    return dep_path


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
    use_gpu: bool = False,
    random_seed: int = 0,
    n_jobs: int | None = None,
    metrics: Sequence[str] | None = None,
    regime_features: Sequence[str] | None = None,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    grad_clip: float = 1.0,
    pretrain_mask: Path | None = None,
    meta_weights: Path | Sequence[float] | None = None,
    hrp_allocation: bool = False,
    strategy_search: bool = False,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
    profile: bool = False,
    controller: AutoMLController | None = None,
    reuse_controller: bool = False,
    complexity_penalty: float = 0.1,
    **kwargs: object,
) -> object:
    """Train a model selected from the registry."""
    chosen_action: tuple[tuple[str, ...], str] | None = None
    if controller is not None:
        if not reuse_controller:
            controller.reset()
        chosen_action, _ = controller.sample_action()
        features = list(chosen_action[0])
        model_type = chosen_action[1]

    set_seed(random_seed)
    configure_cache(
        FeatureConfig(cache_dir=cache_dir, enabled_features=set(features or []))
    )
    if strategy_search:
        from botcopier.strategy.dsl import serialize
        from botcopier.strategy.engine import search_strategies

        prices = np.linspace(1.0, 200.0, 200)
        best, pareto = search_strategies(prices)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.json"
        try:
            existing = json.loads(model_path.read_text())
        except Exception:
            existing = {}
        existing["strategies"] = [
            {
                "expr": serialize(c.expr),
                "return": c.ret,
                "risk": c.risk,
            }
            for c in sorted(pareto, key=lambda x: x.ret, reverse=True)
        ]
        existing["best_strategy"] = serialize(best.expr)
        existing["best_return"] = best.ret
        existing["best_risk"] = best.risk
        model_path.write_text(json.dumps(existing, indent=2))
        return
    tracer = trace.get_tracer(__name__)
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
        "dask",
    ]
    load_kwargs = {k: kwargs[k] for k in load_keys if k in kwargs}
    with tracer.start_as_current_span("data_load"):
        logs, feature_names, data_hashes = _load_logs(data_dir, **load_kwargs)
    logger.info("Training data hashes: %s", data_hashes)

    profiles_dir = out_dir / "profiles"
    if profile:
        profiles_dir.mkdir(parents=True, exist_ok=True)
        feature_prof = cProfile.Profile()
        fit_prof = cProfile.Profile()
        eval_prof = cProfile.Profile()
    else:
        feature_prof = fit_prof = eval_prof = None

    meta_init: np.ndarray | None = None
    meta_info: dict[str, object] | None = None
    if meta_weights is not None:
        if isinstance(meta_weights, (str, Path)):
            try:
                meta_data = json.loads(Path(meta_weights).read_text())
                if isinstance(meta_data.get("meta"), dict):
                    meta_w = meta_data["meta"].get("weights")
                    if meta_w is not None:
                        meta_init = np.asarray(meta_w, dtype=float)
                    meta_info = {
                        k: v for k, v in meta_data["meta"].items() if k != "weights"
                    }
                    meta_info.setdefault("source", str(meta_weights))
                else:
                    meta_w = meta_data.get("meta_weights")
                    if meta_w is not None:
                        meta_init = np.asarray(meta_w, dtype=float)
            except Exception:
                meta_init = None
                meta_info = None
        else:
            meta_init = np.asarray(meta_weights, dtype=float)
    gpu_kwargs: dict[str, object] = {}
    if use_gpu:
        if model_type == "xgboost":
            gpu_kwargs.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
        elif model_type == "catboost":
            gpu_kwargs.update({"device": "gpu"})
        elif model_type == "transformer":
            gpu_kwargs.update({"device": "cuda"})
    fs_repo = Path(__file__).resolve().parents[1] / "feature_store" / "feast_repo"
    span_ctx = tracer.start_as_current_span("feature_extraction")
    span_ctx.__enter__()
    y_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    profit_list: list[np.ndarray] = []
    returns_frames: list[pd.DataFrame] = []
    label_col: str | None = None
    returns_df: pd.DataFrame | None = None
    if profile and feature_prof is not None:
        feature_prof.enable()
    if (
        isinstance(logs, Iterable)
        and not isinstance(logs, (pd.DataFrame,))
        and not (_HAS_POLARS and isinstance(logs, pl.DataFrame))
        and not (_HAS_DASK and isinstance(logs, dd.DataFrame))
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
                chunk, feature_names, _, _ = _extract_features(
                    chunk, feature_names, n_jobs=n_jobs
                )
            FeatureSchema.validate(chunk[feature_names], lazy=True)
            if label_col is None:
                label_col = next(
                    (c for c in chunk.columns if c.startswith("label")), None
                )
                if label_col is None:
                    raise ValueError("no label column found")
            X_list.append(chunk[feature_names].fillna(0.0).to_numpy(dtype=float))
            if "profit" in chunk.columns:
                p = chunk["profit"].to_numpy(dtype=float)
                profit_list.append(p)
                y_list.append((p > 0).astype(float))
                if {"event_time", "symbol"} <= set(chunk.columns):
                    returns_frames.append(chunk[["event_time", "symbol", "profit"]])
            elif label_col is not None:
                y_list.append(chunk[label_col].to_numpy(dtype=float))
            else:
                raise ValueError("no profit or label column found")
        y = np.concatenate(y_list, axis=0)
        X = np.vstack(X_list)
        has_profit = bool(profit_list)
        profits = (
            np.concatenate(profit_list, axis=0) if profit_list else np.zeros_like(y)
        )
        returns_df = (
            pd.concat(returns_frames, ignore_index=True) if returns_frames else None
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
            df, feature_names, _, _ = _extract_features(
                df, feature_names, n_jobs=n_jobs
            )
        FeatureSchema.validate(df[feature_names], lazy=True)
        if _HAS_DASK and isinstance(df, dd.DataFrame):
            df = df.compute()
            X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy(dtype=float)
                y = (profits > 0).astype(float)
                has_profit = True
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy(dtype=float)
                profits = np.zeros_like(y)
                has_profit = False
        elif isinstance(df, pd.DataFrame):
            X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy(dtype=float)
                y = (profits > 0).astype(float)
                has_profit = True
                returns_df = df[["event_time", "symbol", "profit"]]
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy(dtype=float)
                profits = np.zeros_like(y)
                has_profit = False
                returns_df = None
        elif _HAS_POLARS and isinstance(df, pl.DataFrame):
            X = df.select(feature_names).fill_null(0.0).to_numpy().astype(float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy().astype(float)
                y = (profits > 0).astype(float)
                has_profit = True
                returns_df = df[["event_time", "symbol", "profit"]].to_pandas()
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy().astype(float)
                profits = np.zeros_like(y)
                has_profit = False
                returns_df = None
        else:  # pragma: no cover - defensive
            raise TypeError("Unsupported DataFrame type")

    if profile and feature_prof is not None:
        feature_prof.disable()
        feature_prof.dump_stats(str(profiles_dir / "feature_extraction.prof"))
    span_ctx.__exit__(None, None, None)

    if has_profit and profits.size:
        cost = fee_per_trade + np.abs(profits) * slippage_bps * 1e-4
        profits = profits - cost
        y = (profits > 0).astype(float)
    sample_weight = np.clip(profits, a_min=0.0, a_max=None)
    cluster_map: dict[str, list[str]] = {}
    encoder_meta: dict[str, object] | None = None
    regime_feature_names = list(regime_features or [])
    R: np.ndarray | None = None
    if model_type == "moe":
        if not regime_feature_names:
            regime_feature_names = [f for f in feature_names if f.startswith("regime_")]
        if not regime_feature_names:
            raise ValueError("regime_features must be provided for model_type='moe'")
        idx = [feature_names.index(f) for f in regime_feature_names]
        R = X[:, idx]
        X = np.delete(X, idx, axis=1)
        feature_names = [fn for i, fn in enumerate(feature_names) if i not in idx]

    curriculum_threshold = float(kwargs.get("curriculum_threshold", 0.0))
    curriculum_steps = int(kwargs.get("curriculum_steps", 3))
    curriculum_meta: list[dict[str, object]] = []
    if curriculum_threshold > 0.0:
        X, y, profits, sample_weight, R, curriculum_meta = _apply_curriculum(
            X,
            y,
            profits,
            sample_weight,
            model_type=model_type,
            gpu_kwargs=gpu_kwargs,
            grad_clip=grad_clip,
            threshold=curriculum_threshold,
            steps=curriculum_steps,
            R=R,
            regime_feature_names=regime_feature_names or None,
        )

    if pretrain_mask is not None and _HAS_TORCH:
        enc_path = Path(pretrain_mask)
        if enc_path.exists():
            state = torch.load(enc_path, map_location="cpu")
            arch = state.get("architecture", [])
            if arch:
                encoder = torch.nn.Linear(int(arch[0]), int(arch[1]))
                encoder.load_state_dict(state["state_dict"])
                encoder.eval()
                with torch.no_grad():
                    X = encoder(torch.as_tensor(X, dtype=torch.float32)).numpy()
                feature_names = [f"enc_{i}" for i in range(X.shape[1])]
                encoder_meta = {
                    "architecture": arch,
                    "mask_ratio": float(state.get("mask_ratio", 0.0)),
                }

    # --- Power transformation for highly skewed features -----------------
    skew_threshold = float(kwargs.get("skew_threshold", 1.0))
    pt_meta: dict[str, list] | None = None
    if X.size and feature_names:
        df_skew = pd.DataFrame(X, columns=feature_names)
        skewness = df_skew.skew(axis=0).abs()
        skew_cols = skewness[skewness > skew_threshold].index.tolist()
        if skew_cols:
            pt = PowerTransformer(method="yeo-johnson")
            idx = [feature_names.index(c) for c in skew_cols]
            X[:, idx] = pt.fit_transform(X[:, idx])
            pt_meta = {
                "features": skew_cols,
                "lambdas": pt.lambdas_.tolist(),
                "mean": pt._scaler.mean_.tolist(),
                "scale": pt._scaler.scale_.tolist(),
            }

    # --- Correlation-based feature clustering ---------------------------
    corr_thresh = float(kwargs.get("cluster_correlation", 0.9))
    if X.size and feature_names and X.shape[1] >= 2 and corr_thresh < 1.0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        corr = np.corrcoef(X_scaled, rowvar=False)
        dist = 1 - np.abs(corr)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="average")
        cluster_ids = fcluster(link, t=1 - corr_thresh, criterion="distance")
        mi = mutual_info_classif(X_scaled, y)
        keep_idx: list[int] = []
        removed_groups: list[dict[str, list[str] | str]] = []
        for cid in np.unique(cluster_ids):
            idx = np.where(cluster_ids == cid)[0]
            names = [feature_names[i] for i in idx]
            if len(idx) == 1:
                keep_idx.append(idx[0])
                cluster_map[names[0]] = names
                continue
            best_local = idx[np.argmax(mi[idx])]
            rep_name = feature_names[best_local]
            cluster_map[rep_name] = names
            keep_idx.append(best_local)
            dropped = [f for f in names if f != rep_name]
            if dropped:
                removed_groups.append({"kept": rep_name, "dropped": dropped})
        keep_idx = sorted(set(keep_idx))
        if len(keep_idx) < len(feature_names):
            logger.info("Removed correlated feature groups: %s", removed_groups)
            X = X[:, keep_idx]
            feature_names = [feature_names[i] for i in keep_idx]
    else:
        cluster_map = {fn: [fn] for fn in feature_names}

    # --- Baseline statistics for Mahalanobis distance ------------------
    feat_mean: np.ndarray = np.mean(X, axis=0) if X.size else np.array([])
    if X.shape[0] >= 2:
        feat_cov = np.cov(X, rowvar=False)
    else:
        feat_cov = np.eye(X.shape[1]) if X.size else np.empty((0, 0))
    cov_inv = np.linalg.pinv(feat_cov) if feat_cov.size else np.empty((0, 0))
    diff_all = X - feat_mean if X.size else np.empty_like(X)
    mahal_all = (
        np.sqrt(np.einsum("ij,jk,ik->i", diff_all, cov_inv, diff_all))
        if cov_inv.size
        else np.array([])
    )

    mlflow_active = (tracking_uri is not None) or (experiment_name is not None)
    if mlflow_active and not _HAS_MLFLOW:
        raise RuntimeError("mlflow is required for tracking")
    if mlflow_active:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

    span_model = tracer.start_as_current_span("model_fit")
    if profile and fit_prof is not None:
        fit_prof.enable()
    span_model.__enter__()
    run_ctx = mlflow.start_run() if mlflow_active else nullcontext()
    with run_ctx:
        n_splits = int(kwargs.get("n_splits", 3))
        gap = int(kwargs.get("cv_gap", 1))
        splitter = PurgedWalkForward(n_splits=n_splits, gap=gap)
        splits = list(splitter.split(X))
        val_dists = (
            np.concatenate([mahal_all[val_idx] for _, val_idx in splits])
            if mahal_all.size
            else np.array([])
        )
        ood_threshold = (
            float(np.percentile(val_dists, 99)) if val_dists.size else float("inf")
        )
        ood_rate = float(np.mean(val_dists > ood_threshold)) if val_dists.size else 0.0
        param_grid = kwargs.get("param_grid") or [{}]
        metric_names = metrics
        best_score = -np.inf
        best_params: dict[str, object] | None = None
        best_fold_metrics: list[dict[str, object]] = []
        metrics: dict[str, float] = {}
        if distributed and not _HAS_RAY:
            raise RuntimeError("ray is required for distributed execution")
        for params in param_grid:
            fold_metrics: list[dict[str, object]] = []
            if distributed and _HAS_RAY:
                X_ref = ray.put(X)
                y_ref = ray.put(y)
                profits_ref = ray.put(profits)
                weight_ref = ray.put(sample_weight)
                R_ref = ray.put(R) if model_type == "moe" else None

                @ray.remote
                def _run_fold(tr_idx, val_idx, fold):
                    X = ray.get(X_ref)
                    y = ray.get(y_ref)
                    profits = ray.get(profits_ref)
                    weights = ray.get(weight_ref)
                    R_local = ray.get(R_ref) if R_ref is not None else None
                    builder = get_model(model_type)
                    if model_type == "moe" and R_local is not None:
                        model_fold, pred_fn = builder(
                            X[tr_idx],
                            y[tr_idx],
                            regime_features=R_local[tr_idx],
                            regime_feature_names=regime_feature_names,
                            sample_weight=weights[tr_idx],
                            grad_clip=grad_clip,
                            **gpu_kwargs,
                            **params,
                            **(
                                {"init_weights": meta_init}
                                if meta_init is not None
                                else {}
                            ),
                        )
                        prob_val = pred_fn(X[val_idx], R_local[val_idx])
                    else:
                        kwargs = dict(**gpu_kwargs, **params)
                        if model_type != "transformer":
                            kwargs["sample_weight"] = weights[tr_idx]
                        if model_type in {"moe", "transformer"}:
                            kwargs["grad_clip"] = grad_clip
                        if meta_init is not None:
                            kwargs["init_weights"] = meta_init
                        model_fold, pred_fn = builder(
                            X[tr_idx],
                            y[tr_idx],
                            **kwargs,
                        )
                        prob_val = pred_fn(X[val_idx])
                    returns = profits[val_idx] * (prob_val >= 0.5)
                    fold_metric = _classification_metrics(
                        y[val_idx], prob_val, returns, selected=metric_names
                    )
                    fold_metric["max_drawdown"] = _max_drawdown(returns)
                    fold_metric["var_95"] = _var_95(returns)
                    return fold, fold_metric

                futures = [
                    _run_fold.remote(tr_idx, val_idx, fold)
                    for fold, (tr_idx, val_idx) in enumerate(splits)
                    if len(np.unique(y[tr_idx])) >= 2
                ]
                results = ray.get(futures)
                results.sort(key=lambda x: x[0])
                for fold, fold_metric in results:
                    fold_metrics.append(fold_metric)
                    logger.info(
                        "Fold %d params %s metrics %s", fold + 1, params, fold_metric
                    )
                    if mlflow_active:
                        for k, v in fold_metric.items():
                            if isinstance(v, (int, float)) and not np.isnan(v):
                                mlflow.log_metric(f"fold{fold + 1}_{k}", float(v))
            else:
                for fold, (tr_idx, val_idx) in enumerate(splits):
                    if len(np.unique(y[tr_idx])) < 2:
                        continue
                    builder = get_model(model_type)
                    if model_type == "moe" and R is not None:
                        model_fold, pred_fn = builder(
                            X[tr_idx],
                            y[tr_idx],
                            regime_features=R[tr_idx],
                            regime_feature_names=regime_feature_names,
                            sample_weight=sample_weight[tr_idx],
                            grad_clip=grad_clip,
                            **gpu_kwargs,
                            **params,
                            **(
                                {"init_weights": meta_init}
                                if meta_init is not None
                                else {}
                            ),
                        )
                        prob_val = pred_fn(X[val_idx], R[val_idx])
                    else:
                        kwargs = dict(**gpu_kwargs, **params)
                        if model_type != "transformer":
                            kwargs["sample_weight"] = sample_weight[tr_idx]
                        if model_type in {"moe", "transformer"}:
                            kwargs["grad_clip"] = grad_clip
                        if meta_init is not None:
                            kwargs["init_weights"] = meta_init
                        model_fold, pred_fn = builder(
                            X[tr_idx],
                            y[tr_idx],
                            **kwargs,
                        )
                        prob_val = pred_fn(X[val_idx])
                    returns = profits[val_idx] * (prob_val >= 0.5)
                    if profile and eval_prof is not None:
                        eval_prof.enable()
                    fold_metric = _classification_metrics(
                        y[val_idx], prob_val, returns, selected=metric_names
                    )
                    if profile and eval_prof is not None:
                        eval_prof.disable()
                    fold_metric["max_drawdown"] = _max_drawdown(returns)
                    fold_metric["var_95"] = _var_95(returns)
                    fold_metrics.append(fold_metric)
                    logger.info(
                        "Fold %d params %s metrics %s", fold + 1, params, fold_metric
                    )
                    if mlflow_active:
                        for k, v in fold_metric.items():
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
            risk_penalty = 0.0
            if max_drawdown is not None:
                risk_penalty += max(0.0, agg.get("max_drawdown", 0.0) - max_drawdown)
            if var_limit is not None:
                risk_penalty += max(0.0, agg.get("var_95", 0.0) - var_limit)
            score -= risk_penalty
            if score > best_score:
                best_score = score
                best_params = params
                best_fold_metrics = fold_metrics
                metrics = agg
        if profile and eval_prof is not None:
            eval_prof.dump_stats(str(profiles_dir / "evaluation.prof"))
        metrics["ood_rate"] = ood_rate
        min_acc = float(kwargs.get("min_accuracy", 0.0))
        min_profit = float(kwargs.get("min_profit", -np.inf))
        if metrics and (
            metrics.get("accuracy", 0.0) < min_acc
            or metrics.get("profit", 0.0) < min_profit
        ):
            raise ValueError("Cross-validation metrics below thresholds")
        builder = get_model(model_type)
        if model_type == "moe" and R is not None:
            model_data, predict_fn = builder(
                X,
                y,
                regime_features=R,
                regime_feature_names=regime_feature_names,
                sample_weight=sample_weight,
                grad_clip=grad_clip,
                **gpu_kwargs,
                **(best_params or {}),
                **({"init_weights": meta_init} if meta_init is not None else {}),
            )
        else:
            kwargs = dict(**gpu_kwargs, **(best_params or {}))
            if model_type != "transformer":
                kwargs["sample_weight"] = sample_weight
            if model_type in {"moe", "transformer"}:
                kwargs["grad_clip"] = grad_clip
            if meta_init is not None:
                kwargs["init_weights"] = meta_init
            model_data, predict_fn = builder(
                X,
                y,
                **kwargs,
            )

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
                        kwargs = dict(**gpu_kwargs, **(best_params or {}))
                        if model_type != "transformer":
                            kwargs["sample_weight"] = sample_weight
                        if model_type in {"moe", "transformer"}:
                            kwargs["grad_clip"] = grad_clip
                        model_data, predict_fn = builder(
                            X,
                            y,
                            **kwargs,
                        )
        except Exception:  # pragma: no cover - shap is optional
            logger.exception("Failed to compute SHAP values")
        model_obj = getattr(predict_fn, "model", None)

        # --- Probability calibration ---------------------------------------
        calibration_info: dict[str, object] | None = None
        if model_type != "moe":
            base_model = getattr(predict_fn, "model", None)
            if base_model is not None and X.size and y.size:
                cal_splitter = PurgedWalkForward(n_splits=n_splits, gap=gap)
                calibrator = CalibratedClassifierCV(
                    base_model, method="isotonic", cv=cal_splitter
                )
                fit_kwargs = (
                    {"sample_weight": sample_weight} if sample_weight.size else {}
                )
                try:
                    calibrator.fit(X, y, **fit_kwargs)
                except ValueError:
                    try:
                        calibrator = CalibratedClassifierCV(
                            base_model, method="isotonic", cv="prefit"
                        )
                        calibrator.fit(X, y, **fit_kwargs)
                    except ValueError:
                        calibrator = None

                if calibrator is not None:

                    def _calibrated_predict(arr: np.ndarray) -> np.ndarray:
                        return calibrator.predict_proba(arr)[:, 1]

                    _calibrated_predict.model = calibrator  # type: ignore[attr-defined]
                    predict_fn = _calibrated_predict
                    try:
                        iso = calibrator.calibrated_classifiers_[0].calibrators_[0]
                        calibration_info = {
                            "method": "isotonic",
                            "x": iso.X_thresholds_.tolist(),
                            "y": iso.y_thresholds_.tolist(),
                        }
                    except Exception:
                        calibration_info = None
            model_obj = base_model

        if profile and fit_prof is not None:
            fit_prof.disable()
            fit_prof.dump_stats(str(profiles_dir / "model_fit.prof"))
        span_model.__exit__(None, None, None)
        span_eval = tracer.start_as_current_span("evaluation")
        span_eval.__enter__()
        if model_type == "moe" and R is not None:
            probas = predict_fn(X, R)
        else:
            probas = predict_fn(X)
        preds = (probas >= 0.5).astype(float)
        score = float((preds == y).mean())
        feature_metadata = [FeatureMetadata(original_column=fn) for fn in feature_names]
        model = {
            "feature_names": feature_names,
            "feature_metadata": feature_metadata,
            **model_data,
            "model_type": model_type,
        }
        if meta_init is not None:
            meta_entry = {"weights": meta_init.tolist()}
            if meta_info:
                meta_entry.update(meta_info)
            meta_entry["adapted"] = True
            model.setdefault("meta", meta_entry)
            model.setdefault("meta_weights", meta_init.tolist())
        if encoder_meta is not None:
            model["masked_encoder"] = encoder_meta
        if getattr(technical_features, "_DEPTH_CNN_STATE", None) is not None:
            model["depth_cnn"] = technical_features._DEPTH_CNN_STATE
        if getattr(technical_features, "_CSD_PARAMS", None) is not None:
            model["csd_params"] = technical_features._CSD_PARAMS
        if getattr(technical_features, "_GRAPH_SNAPSHOT", None) is not None:
            model["graph_snapshot"] = technical_features._GRAPH_SNAPSHOT
        if cluster_map:
            model["feature_clusters"] = cluster_map
        if calibration_info is not None:
            model["calibration"] = calibration_info
        if pt_meta is not None:
            keep = [f for f in pt_meta["features"] if f in feature_names]
            if keep:
                idx = [pt_meta["features"].index(f) for f in keep]
                pt_meta = {
                    "features": keep,
                    "lambdas": [pt_meta["lambdas"][i] for i in idx],
                    "mean": [pt_meta["mean"][i] for i in idx],
                    "scale": [pt_meta["scale"][i] for i in idx],
                }
                model["power_transformer"] = pt_meta
        if "feature_mean" not in model:
            model["feature_mean"] = X.mean(axis=0).tolist()
        if "feature_std" not in model:
            model["feature_std"] = X.std(axis=0).tolist()
        if "clip_low" not in model:
            model["clip_low"] = np.min(X, axis=0).tolist()
        if "clip_high" not in model:
            model["clip_high"] = np.max(X, axis=0).tolist()
        model["ood"] = {
            "mean": feat_mean.tolist(),
            "covariance": feat_cov.tolist(),
            "threshold": ood_threshold,
        }
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
        model["cv_accuracy"] = metrics.get("accuracy", 0.0)
        model["cv_profit"] = metrics.get("profit", 0.0)
        model["conformal_lower"] = 0.0
        model["conformal_upper"] = 1.0
        model["session_models"]["asian"]["cv_metrics"] = best_fold_metrics
        model["session_models"]["asian"]["conformal_lower"] = 0.0
        model["session_models"]["asian"]["conformal_upper"] = 1.0
        mode = kwargs.get("mode")
        if mode is not None:
            model["mode"] = mode
        model["risk_params"] = {
            "max_drawdown": max_drawdown,
            "var_limit": var_limit,
        }
        model["risk_metrics"] = {
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "var_95": metrics.get("var_95", 0.0),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        model.setdefault("metadata", {})["seed"] = random_seed
        model["data_hashes"] = data_hashes
        if curriculum_meta:
            model["curriculum"] = curriculum_meta
            # Record summary of the final curriculum phase for quick access
            model["curriculum_final"] = curriculum_meta[-1]
        if model_obj is not None:
            try:
                from botcopier.scripts.explain_model import generate_explanations

                report_dir = out_dir / "reports" / "explanations"
                report_path = report_dir / "explanation.md"
                generate_explanations(model_obj, X, y, feature_names, report_path)
                model["explanation_report"] = str(report_path.relative_to(out_dir))
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to generate explanation report")
        if hrp_allocation and returns_df is not None:
            try:
                pivot = returns_df.pivot_table(
                    index="event_time", columns="symbol", values="profit", aggfunc="sum"
                ).fillna(0.0)
                if pivot.shape[1] >= 1:
                    weights, link = hierarchical_risk_parity(pivot)
                    model["hrp_weights"] = weights.to_dict()
                    model["hrp_dendrogram"] = link.tolist()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to compute HRP allocation")
        model_params = ModelParams(**model)
        (out_dir / "model.json").write_text(model_params.model_dump_json())
        (out_dir / "data_hashes.json").write_text(json.dumps(data_hashes, indent=2))
        if controller is not None and chosen_action is not None:
            profit = metrics.get("profit", 0.0)
            subset, model_choice = chosen_action
            complexity = len(subset) + controller.models.get(model_choice, 0)
            risk_penalty = 0.0
            if max_drawdown is not None:
                risk_penalty += max(
                    0.0, metrics.get("max_drawdown", 0.0) - max_drawdown
                )
            if var_limit is not None:
                risk_penalty += max(0.0, metrics.get("var_95", 0.0) - var_limit)
            reward = profit - complexity_penalty * complexity - risk_penalty
            controller.update(chosen_action, reward)
        deps_file = _write_dependency_snapshot(out_dir)
        generate_model_card(
            model_params,
            metrics,
            out_dir / "model_card.md",
            dependencies_path=deps_file,
        )
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
            for k, v in metrics.items():
                mlflow.log_metric(f"cv_{k}", float(v))
            mlflow.log_artifact(str(out_dir / "model.json"), artifact_path="model")
            mlflow.log_artifact(
                str(out_dir / "data_hashes.json"), artifact_path="model"
            )
        span_eval.__exit__(None, None, None)
        return model_obj


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
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="enable GPU acceleration for supported models",
    )
    p.add_argument("--random-seed", dest="random_seed", type=int, default=0)
    p.add_argument("--trace", action="store_true", help="Enable OpenTelemetry tracing")
    p.add_argument(
        "--trace-exporter",
        choices=["otlp", "jaeger"],
        default="otlp",
        help="Tracing exporter to use",
    )
    p.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="classification metric to compute (repeatable)",
    )
    p.add_argument(
        "--grad-clip",
        dest="grad_clip",
        type=float,
        default=1.0,
        help="max gradient norm for PyTorch models",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Profile feature extraction, model fitting, and evaluation",
    )
    p.add_argument(
        "--strategy-search",
        action="store_true",
        help="Run DSL strategy search before training",
    )
    p.add_argument(
        "--reuse-controller",
        action="store_true",
        help="Reuse saved AutoML controller policy if available",
    )
    p.add_argument(
        "--use-meta",
        type=Path,
        dest="use_meta",
        help="Path to model.json with meta-weights",
    )
    args = p.parse_args()
    setup_logging(enable_tracing=args.trace, exporter=args.trace_exporter)
    cfg = TrainingConfig(random_seed=args.random_seed)
    set_seed(cfg.random_seed)
    train(
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        use_gpu=args.use_gpu,
        random_seed=cfg.random_seed,
        metrics=args.metrics,
        grad_clip=args.grad_clip,
        profile=args.profile,
        strategy_search=args.strategy_search,
        reuse_controller=args.reuse_controller,
        meta_weights=args.use_meta,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
