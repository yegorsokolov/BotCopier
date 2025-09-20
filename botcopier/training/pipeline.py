"""Training pipeline orchestrating model training and evaluation."""

from __future__ import annotations

import argparse
import cProfile
import gzip
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
from joblib import Memory
from opentelemetry import trace
from pydantic import ValidationError
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler

import botcopier.features.technical as technical_features
from automl.controller import AutoMLController
from botcopier.config.settings import (
    DataConfig,
    ExecutionConfig,
    TrainingConfig,
    load_settings,
)
from botcopier.data.feature_schema import FeatureSchema
from botcopier.data.loading import _load_logs
from botcopier.exceptions import TrainingPipelineError
from botcopier.features.anomaly import _clip_train_features
from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    _neutralize_against_market_index,
    configure_cache,
)
from botcopier.models.registry import MODEL_REGISTRY, get_model, load_params
from botcopier.models.schema import FeatureMetadata, ModelParams
from botcopier.scripts.evaluation import (
    _classification_metrics,
    search_decision_threshold,
)
from botcopier.scripts.model_card import generate_model_card
from botcopier.scripts.portfolio import hierarchical_risk_parity
from botcopier.scripts.splitters import PurgedWalkForward
from botcopier.shap_utils import mean_absolute_shap
from botcopier.training.curriculum import _apply_curriculum
from botcopier.training.evaluation import (
    HAS_OPTUNA,
    HAS_RAY,
    max_drawdown as eval_max_drawdown,
    optuna,
    ray,
    resolve_data_path,
    serialise_metric_values,
    suggest_model_params,
    trial_logger,
    var_95 as eval_var_95,
)
from botcopier.training.preprocessing import (
    HAS_DASK,
    HAS_FEAST,
    HAS_POLARS,
    HAS_TORCH,
    FEATURE_COLUMNS,
    FeatureStore,
    apply_autoencoder_from_metadata,
    autoencoder_metadata_path,
    filter_feature_matrix,
    encode_with_autoencoder,
    load_autoencoder_metadata,
    load_news_embeddings,
    normalise_feature_subset,
    save_autoencoder_metadata,
    should_use_lightweight,
    train_lightweight,
    dd,
    pl,
    torch,
)
from botcopier.training.sequence_builders import (
    build_window_sequences,
    prepare_symbol_context,
)
from botcopier.training.tracking import (
    HAS_DVC,
    HAS_MLFLOW,
    DvcException,
    DvcRepo,
    mlflow,
    serialize_mlflow_param,
    version_artifacts_with_dvc,
    write_dependency_snapshot,
)
from botcopier.training.weighting import (
    build_sample_weights,
    normalise_weights,
    summarise_weights,
)
from botcopier.utils.random import set_seed
from logging_utils import setup_logging
from metrics.aggregator import add_metric

logger = logging.getLogger(__name__)

SEQUENCE_MODEL_TYPES = {"tabtransformer", "tcn", "crossmodal"}


def run_optuna(
    n_trials: int = 10,
    csv_path: Path | str = "hyperparams.csv",
    model_json_path: Path | str = "model.json",
    *,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    sampler: "optuna.samplers.BaseSampler | None" = None,
    settings_overrides: Mapping[str, Any] | None = None,
    config_path: Path | str = Path("params.yaml"),
    model_types: Sequence[str] | None = None,
    feature_flags: Mapping[str, Sequence[bool]] | None = None,
    train_kwargs: Mapping[str, Any] | None = None,
) -> optuna.study.Study:
    """Run an Optuna study using the real training pipeline."""

    if not HAS_OPTUNA:  # pragma: no cover - defensive
        raise RuntimeError("optuna is required to run hyperparameter optimisation")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_json_path = Path(model_json_path)
    model_json_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)

    overrides = dict(settings_overrides or {})
    data_cfg, train_cfg, exec_cfg = load_settings(overrides, path=config_path)
    data_path = resolve_data_path(data_cfg)
    study_out_dir = Path(data_cfg.out_dir) if data_cfg.out_dir else model_json_path.parent
    study_out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = study_out_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    sampler = sampler or optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(study_name and storage),
    )

    base_train_kwargs: dict[str, Any] = {
        "model_type": train_cfg.model_type,
        "cache_dir": train_cfg.cache_dir,
        "tracking_uri": train_cfg.tracking_uri,
        "experiment_name": train_cfg.experiment_name,
        "features": list(train_cfg.features) if train_cfg.features else None,
        "regime_features": (
            list(train_cfg.regime_features) if train_cfg.regime_features else None
        ),
        "use_gpu": exec_cfg.use_gpu,
        "metrics": list(train_cfg.metrics) if train_cfg.metrics else None,
        "grad_clip": train_cfg.grad_clip,
        "meta_weights": train_cfg.meta_weights,
        "hrp_allocation": train_cfg.hrp_allocation,
        "strategy_search": train_cfg.strategy_search,
        "max_drawdown": max_drawdown,
        "var_limit": var_limit,
        "vol_weight": train_cfg.vol_weight,
        "profile": exec_cfg.profile,
        "reuse_controller": train_cfg.reuse_controller,
        "controller_max_subset_size": train_cfg.controller_max_subset_size,
        "controller_episode_sample_size": train_cfg.controller_episode_sample_size,
        "controller_episode_combination_cap": train_cfg.controller_episode_combination_cap,
        "controller_baseline_momentum": train_cfg.controller_baseline_momentum,
        "random_seed": train_cfg.random_seed,
    }
    if train_cfg.half_life_days:
        base_train_kwargs["half_life_days"] = train_cfg.half_life_days
    if train_kwargs:
        base_train_kwargs.update(train_kwargs)

    # Filter out None entries to avoid overriding defaults inside ``train``
    base_train_kwargs = {
        key: value for key, value in base_train_kwargs.items() if value is not None
    }
    base_param_grid = base_train_kwargs.pop("param_grid", None)

    model_choices = list(model_types or [])
    if model_choices:
        invalid = [m for m in model_choices if m not in MODEL_REGISTRY]
        if invalid:
            raise ValueError(f"Unknown model types for optuna search: {invalid}")
    flag_options = {
        name: tuple(options)
        for name, options in (feature_flags or {}).items()
        if options
    }

    def _objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        trial_dir = trials_dir / f"trial_{trial.number}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_kwargs = dict(base_train_kwargs)
        if model_choices:
            model_type = trial.suggest_categorical(
                "model_type", sorted(set(model_choices))
            )
        else:
            model_type = str(trial_kwargs.get("model_type", train_cfg.model_type))
        trial_kwargs["model_type"] = model_type

        for flag_name, options in flag_options.items():
            value = trial.suggest_categorical(flag_name, list(options))
            trial_kwargs[flag_name] = value

        seed = trial.suggest_int("seed", 0, 9999)
        trial_kwargs["random_seed"] = seed
        trial.set_user_attr("seed", seed)

        if "half_life_days" not in base_train_kwargs:
            trial_kwargs["half_life_days"] = trial.suggest_float(
                "half_life_days", 0.0, 30.0
            )
        model_params = suggest_model_params(trial, model_type)
        if model_params:
            trial_kwargs["param_grid"] = [model_params]
        elif base_param_grid is not None:
            trial_kwargs["param_grid"] = [dict(p) for p in base_param_grid]
        else:
            trial_kwargs["param_grid"] = [{}]

        try:
            train(data_path, trial_dir, **trial_kwargs)
        except Exception as exc:  # pragma: no cover - surfaced through optuna
            trial.set_user_attr("exception", repr(exc))
            raise

        model_path = trial_dir / "model.json"
        model_data = json.loads(model_path.read_text())
        risk_metrics = model_data.get("risk_metrics", {})
        metrics = model_data.get("cv_metrics", {})
        profit = float(model_data.get("cv_profit", metrics.get("profit", 0.0) or 0.0))
        sharpe = float(
            risk_metrics.get("sharpe_ratio")
            or metrics.get("sharpe_ratio")
            or metrics.get("sharpe")
            or 0.0
        )
        drawdown = float(
            risk_metrics.get("max_drawdown")
            or metrics.get("max_drawdown")
            or 0.0
        )
        var95 = float(risk_metrics.get("var_95") or metrics.get("var_95") or 0.0)

        trial.set_user_attr("artifact_dir", str(trial_dir))
        trial.set_user_attr("model_path", str(model_path))
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr(
            "metrics",
            serialise_metric_values(
                {
                    "cv_profit": profit,
                    "risk_metrics": risk_metrics,
                    "cv_metrics": metrics,
                }
            ),
        )
        trial.set_user_attr("max_drawdown", drawdown)
        trial.set_user_attr("var_95", var95)

        return profit, sharpe, drawdown

    study.optimize(
        _objective,
        n_trials=n_trials,
        callbacks=[trial_logger(csv_path)],
        catch=(Exception,),
    )

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    if not completed:
        raise RuntimeError("No successful optuna trials completed")

    candidates = [t for t in study.best_trials if t in completed]
    if not candidates:
        candidates = completed
    if max_drawdown is not None:
        candidates = [t for t in candidates if t.values[2] <= max_drawdown]
    if var_limit is not None:
        candidates = [
            t for t in candidates if t.user_attrs.get("var_95", float("inf")) <= var_limit
        ]
    if not candidates:
        candidates = completed
    best = max(candidates, key=lambda t: (t.values[0], t.values[1]))

    final_kwargs = dict(base_train_kwargs)
    final_kwargs.pop("param_grid", None)
    model_params_best = best.user_attrs.get("model_params") or {}
    if model_params_best:
        final_kwargs["param_grid"] = [model_params_best]
    elif base_param_grid is not None:
        final_kwargs["param_grid"] = [dict(p) for p in base_param_grid]
    else:
        final_kwargs["param_grid"] = [{}]
    if model_choices and "model_type" in best.params:
        final_kwargs["model_type"] = best.params["model_type"]
    if "half_life_days" in best.params:
        final_kwargs["half_life_days"] = best.params["half_life_days"]
    for flag_name in flag_options:
        if flag_name in best.params:
            final_kwargs[flag_name] = best.params[flag_name]
    if "seed" in best.params:
        final_kwargs["random_seed"] = int(best.params["seed"])

    train(data_path, study_out_dir, **final_kwargs)

    final_model_path = study_out_dir / "model.json"
    model_data = json.loads(final_model_path.read_text())
    metadata = model_data.setdefault("metadata", {})
    relative_csv = os.path.relpath(csv_path, study_out_dir)
    metadata["hyperparam_log"] = relative_csv
    selected_trial = {
        "number": best.number,
        "profit": float(best.values[0]),
        "sharpe": float(best.values[1]),
        "max_drawdown": float(best.values[2]),
        "var_95": float(best.user_attrs.get("var_95", 0.0)),
        "search_params": serialise_metric_values(best.params),
        "model_params": serialise_metric_values(best.user_attrs.get("model_params", {})),
    }
    metadata["selected_trial"] = selected_trial
    metadata["hyperparameter_optimization"] = {
        "study_name": study.study_name,
        "n_trials": len(completed),
        "best_trial": {
            "number": best.number,
            "values": {
                "profit": float(best.values[0]),
                "sharpe": float(best.values[1]),
                "max_drawdown": float(best.values[2]),
                "var_95": float(best.user_attrs.get("var_95", 0.0)),
            },
            "params": serialise_metric_values(best.params),
            "model_params": serialise_metric_values(
                best.user_attrs.get("model_params", {})
            ),
            "artifact_dir": os.path.relpath(
                best.user_attrs.get("artifact_dir", study_out_dir), study_out_dir
            ),
        },
    }

    updated = ModelParams(**model_data)
    final_model_path.write_text(updated.model_dump_json())
    if model_json_path != final_model_path:
        model_json_path.write_text(final_model_path.read_text())
        for name in ("data_hashes.json", "dependencies.txt", "config_snapshot.json"):
            src = study_out_dir / name
            if src.exists():
                dst = model_json_path.parent / name
                if dst != src:
                    dst.write_text(src.read_text())

    return study


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    model_type: str = "logreg",
    cache_dir: Path | None = None,
    model_json: Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    features: Sequence[str] | None = None,
    feature_subset: Sequence[str] | None = None,
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
    half_life_days: float | None = None,
    vol_weight: bool = False,
    profile: bool = False,
    controller: AutoMLController | None = None,
    reuse_controller: bool = False,
    controller_max_subset_size: int | None = None,
    controller_episode_sample_size: int | None = None,
    controller_episode_combination_cap: int | None = None,
    controller_baseline_momentum: float | None = None,
    complexity_penalty: float = 0.1,
    dvc_repo: Path | str | None = None,
    config_hash: str | None = None,
    config_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
    **kwargs: object,
) -> object:
    """Train a model selected from the registry.

    Parameters
    ----------
    dvc_repo
        Optional path to a DVC repository used to version the training
        dataset and resulting model artifacts.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    start_time = datetime.now(UTC)
    run_identifier = uuid4().hex
    config_snapshot_path: Path | None = None
    subset_override = kwargs.pop("feature_subset", None)
    if feature_subset is None and subset_override is not None:
        subset_source = subset_override
    else:
        subset_source = feature_subset
    selected_kwarg = kwargs.pop("selected_features", None)
    if subset_source is None and selected_kwarg is not None:
        subset_source = selected_kwarg
    if subset_source is None:
        user_subset: list[str] = []
        subset_provided = False
    else:
        user_subset, subset_provided = normalise_feature_subset(subset_source)
    active_subset: list[str] = list(user_subset)
    extra_options = cast(dict[str, Any], dict(kwargs))
    force_heavy = bool(extra_options.pop("force_heavy", False))
    if pretrain_mask is None and not force_heavy and should_use_lightweight(data_dir, extra_options):
        return train_lightweight(
            data_dir,
            out_dir,
            extra_prices=cast(
                Mapping[str, Sequence[float]] | None, extra_options.get("extra_prices")
            ),
            config_hash=config_hash,
            config_snapshot=config_snapshot,
            feature_subset=active_subset if active_subset else None,
        )
    if dvc_repo is not None:
        dvc_repo_path: Path | None = Path(dvc_repo)
    else:
        env_repo = os.getenv("BOTCOPIER_DVC_ROOT")
        dvc_repo_path = Path(env_repo) if env_repo else None
    if dvc_repo_path is not None and not dvc_repo_path.exists():
        logger.debug("Provided DVC repository %s does not exist", dvc_repo_path)
        dvc_repo_path = None

    if model_json is not None:
        model_json = Path(model_json)
    controller_kwargs: dict[str, object] = {}
    if controller_max_subset_size is not None:
        controller_kwargs["max_subset_size"] = controller_max_subset_size
    if controller_episode_sample_size is not None:
        controller_kwargs["episode_sample_size"] = controller_episode_sample_size
    if controller_episode_combination_cap is not None:
        controller_kwargs["episode_combination_cap"] = controller_episode_combination_cap
    if controller_baseline_momentum is not None:
        controller_kwargs["baseline_momentum"] = controller_baseline_momentum

    chosen_action: tuple[tuple[str, ...], str] | None = None
    if controller is not None:
        controller.model_path = out_dir / "model.json"
        if controller_kwargs:
            controller.configure(**controller_kwargs)
        if not reuse_controller:
            controller.reset()
        chosen_action, _ = controller.sample_action()
        controller_subset, _ = normalise_feature_subset(chosen_action[0])
        telemetry_payload: dict[str, object] = dict(controller.telemetry)
        telemetry_payload.update(
            {
                "selected_features": list(controller_subset),
                "selected_model": chosen_action[1],
            }
        )
        logger.info("Controller telemetry: %s", telemetry_payload)
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            for key, value in telemetry_payload.items():
                if isinstance(value, (bool, int, float)):
                    current_span.set_attribute(f"controller.{key}", value)
        add_metric("controller_episode", telemetry_payload)
        if subset_provided:
            allowed = set(user_subset)
            invalid = [name for name in controller_subset if name not in allowed]
            if invalid:
                raise ValueError(
                    "Controller selected features outside the allowed subset: %s"
                    % invalid
                )
        if controller_subset:
            active_subset = list(controller_subset)
            subset_provided = True
        features = list(controller_subset)
        model_type = chosen_action[1]

    filtered_feature_subset: list[str] | None = None

    if model_type == "transformer":
        logger.warning(
            "Model type 'transformer' is deprecated; using 'tabtransformer' instead"
        )
        model_type = "tabtransformer"
    sequence_model = model_type in SEQUENCE_MODEL_TYPES
    sequence_param_map: dict[str, set[str]] = {
        "tabtransformer": {
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "dim",
            "depth",
            "heads",
            "ff_dim",
            "patience",
            "mixed_precision",
        },
        "tcn": {
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "channels",
            "kernel_size",
            "patience",
            "mixed_precision",
        },
        "crossmodal": {
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "dim",
            "depth",
            "heads",
            "ff_dim",
            "patience",
            "mixed_precision",
        },
        "multi_symbol": {
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "hidden_dim",
            "heads",
            "patience",
            "mixed_precision",
        },
        "moe": {
            "epochs",
            "lr",
            "n_experts",
            "dropout",
        },
    }
    extra_model_params = {
        key: kwargs[key]
        for key in sequence_param_map.get(model_type, set())
        if key in kwargs
    }

    half_life_value = half_life_days
    if half_life_value is None:
        half_life_value = kwargs.pop("half_life", None)
    if half_life_value is None and "half_life_days" in kwargs:
        half_life_value = kwargs.pop("half_life_days")
    half_life_days = float(half_life_value or 0.0)

    if not vol_weight and "vol_weight" in kwargs:
        vol_weight = bool(kwargs.pop("vol_weight"))
    if not vol_weight and "volatility_weighting" in kwargs:
        vol_weight = bool(kwargs.pop("volatility_weighting"))

    threshold_objective = str(
        kwargs.pop("threshold_objective", "profit") or "profit"
    ).lower()
    threshold_grid_param = kwargs.pop("threshold_grid", None)
    threshold_grid_values: np.ndarray | None
    if threshold_grid_param is None:
        threshold_grid_values = None
    elif isinstance(threshold_grid_param, (float, int)):
        threshold_grid_values = np.asarray([float(threshold_grid_param)], dtype=float)
    else:
        try:
            threshold_grid_values = np.asarray(list(threshold_grid_param), dtype=float)
        except TypeError:
            threshold_grid_values = np.asarray(
                [float(threshold_grid_param)], dtype=float
            )
    set_seed(random_seed)
    feature_config = configure_cache(
        FeatureConfig(
            cache_dir=cache_dir,
            enabled_features=set(features or []),
            n_jobs=n_jobs,
        )
    )
    memory = Memory(str(cache_dir) if cache_dir else None, verbose=0)
    _classification_metrics_cached = memory.cache(_classification_metrics)
    _hrp_cached = memory.cache(hierarchical_risk_parity)
    if strategy_search:
        from botcopier.strategy.dsl import serialize
        from botcopier.strategy.search import search_strategies

        def _load_price_series(root: Path) -> np.ndarray | None:
            if not root.exists():
                return None
            for candidate in sorted(root.glob("*.npy")):
                try:
                    arr = np.load(candidate)
                except Exception:
                    continue
                arr = np.asarray(arr, dtype=float).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size >= 2:
                    return arr
            for candidate in sorted(root.glob("*.csv")):
                try:
                    df = pd.read_csv(candidate)
                except Exception:
                    continue
                numeric = df.select_dtypes(include=[np.number])
                if numeric.empty:
                    continue
                series = numeric.iloc[:, 0].to_numpy(dtype=float)
                series = series[np.isfinite(series)]
                if series.size >= 2:
                    return series
            return None

        prices = _load_price_series(data_dir)
        if prices is None or prices.size < 2:
            rng = np.random.default_rng(random_seed or 0)
            steps = rng.normal(loc=0.15, scale=1.0, size=512)
            prices = np.cumsum(steps) + 100.0
            prices = prices - np.nanmin(prices) + 1.0
        prices = np.asarray(prices, dtype=float)

        base_pop = max(8, prices.size // 8)
        population_size = int(min(48, max(12, base_pop)))
        n_generations = int(max(6, np.ceil(160 / max(population_size, 1))))
        n_samples = population_size * n_generations
        search_seed = int(random_seed or 0)

        result = search_strategies(
            prices,
            seed=search_seed,
            population_size=population_size,
            n_generations=n_generations,
            n_samples=n_samples,
        )
        best = result.best
        pareto = result.pareto
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.json"
        try:
            existing = json.loads(model_path.read_text())
        except Exception:
            existing = {}
        existing["strategies"] = [
            {
                "expr": serialize(candidate.expr),
                "return": candidate.ret,
                "risk": candidate.risk,
                "complexity": candidate.complexity,
            }
            for candidate in pareto
        ]
        existing["best_strategy"] = serialize(best.expr)
        existing["best_return"] = best.ret
        existing["best_risk"] = best.risk
        existing["best_complexity"] = best.complexity
        metadata = {
            "seed": search_seed,
            "population_size": population_size,
            "n_generations": n_generations,
            "n_samples": n_samples,
            "price_history": {
                "length": int(prices.size),
                "min": float(np.nanmin(prices)),
                "max": float(np.nanmax(prices)),
            },
        }
        metadata.update(result.metadata)
        existing["strategy_search_metadata"] = metadata
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
    load_kwargs.setdefault("feature_config", feature_config)
    with tracer.start_as_current_span("data_load"):
        logs, feature_names, data_hashes = _load_logs(data_dir, **load_kwargs)
    news_embeddings_df, news_hashes = load_news_embeddings(data_dir)
    if news_hashes:
        data_hashes.update({k: v for k, v in news_hashes.items() if v is not None})
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
        elif sequence_model:
            gpu_kwargs.update({"device": "cuda"})
    fs_repo = Path(__file__).resolve().parents[1] / "feature_store" / "feast_repo"
    span_ctx = tracer.start_as_current_span("feature_extraction")
    span_ctx.__enter__()
    news_window_param = int(kwargs.get("news_window", kwargs.get("window", 5) or 5))
    news_window_param = max(1, news_window_param)
    news_horizon_seconds = float(kwargs.get("news_horizon_seconds", 3600.0))
    news_seq_list: list[np.ndarray] = []
    news_meta: dict[str, object] | None = None
    news_sequences: np.ndarray | None = None
    y_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    profit_list: list[np.ndarray] = []
    event_time_list: list[np.ndarray] = []
    returns_frames: list[pd.DataFrame] = []
    event_times: np.ndarray = np.array([], dtype="datetime64[ns]")
    label_col: str | None = None
    returns_df: pd.DataFrame | None = None
    symbol_batches: list[np.ndarray] = []
    symbols: np.ndarray = np.array([], dtype=str)
    feature_extra_kwargs = {
        key: kwargs[key]
        for key in (
            "symbol_graph",
            "calendar_file",
            "event_window",
            "news_sentiment",
            "entity_graph",
            "neighbor_corr_windows",
            "regime_model",
            "tick_encoder",
            "depth_cnn",
            "calendar_features",
            "pca_components",
            "gnn_state",
            "rank_features",
        )
        if key in kwargs
    }
    if profile and feature_prof is not None:
        feature_prof.enable()
    if (
        isinstance(logs, Iterable)
        and not isinstance(logs, (pd.DataFrame,))
        and not (HAS_POLARS and isinstance(logs, pl.DataFrame))
        and not (HAS_DASK and isinstance(logs, dd.DataFrame))
    ):
        store = FeatureStore(repo_path=str(fs_repo)) if HAS_FEAST else None
        feature_refs = [f"trade_features:{f}" for f in FEATURE_COLUMNS]
        for chunk in logs:
            if HAS_FEAST and store is not None:
                feat_df = store.get_historical_features(
                    entity_df=chunk, features=feature_refs
                ).to_df()
                chunk = chunk.merge(feat_df, on=["symbol", "event_time"], how="left")
                feature_names = list(FEATURE_COLUMNS)
            else:
                chunk, feature_names, _, _ = _extract_features(
                    chunk,
                    feature_names,
                    n_jobs=n_jobs,
                    model_json=model_json,
                    news_embeddings=news_embeddings_df,
                    news_embedding_window=news_window_param,
                    news_embedding_horizon=news_horizon_seconds,
                    config=feature_config,
                    **feature_extra_kwargs,
                )
            meta_entry = technical_features._FEATURE_METADATA.get("__news_embeddings__")
            if meta_entry is not None:
                seq_arr = np.array(meta_entry.get("sequences", []), dtype=float, copy=True)
                if seq_arr.size:
                    news_seq_list.append(seq_arr)
                    if news_meta is None:
                        news_meta = {
                            key: meta_entry.get(key)
                            for key in (
                                "window",
                                "dimension",
                                "columns",
                                "horizon_seconds",
                            )
                        }
            FeatureSchema.validate(chunk[feature_names], lazy=True)
            if label_col is None:
                label_col = next(
                    (c for c in chunk.columns if c.startswith("label")), None
                )
                if label_col is None:
                    raise ValueError("no label column found")
            X_list.append(chunk[feature_names].fillna(0.0).to_numpy(dtype=float))
            if "symbol" in chunk.columns:
                symbol_batches.append(chunk["symbol"].astype(str).to_numpy())
            if "event_time" in chunk.columns:
                event_time_list.append(
                    pd.to_datetime(chunk["event_time"], errors="coerce").to_numpy()
                )
            else:
                event_time_list.append(
                    np.full(len(chunk), np.datetime64("NaT"), dtype="datetime64[ns]")
                )
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
        event_times = (
            pd.to_datetime(np.concatenate(event_time_list), errors="coerce").to_numpy()
            if event_time_list
            else np.array([], dtype="datetime64[ns]")
        )
        if symbol_batches:
            symbols = np.concatenate(symbol_batches)
        else:
            symbols = np.array([], dtype=str)
    else:
        df = logs  # type: ignore[assignment]
        if HAS_FEAST:
            store = FeatureStore(repo_path=str(fs_repo))
            feature_refs = [f"trade_features:{f}" for f in FEATURE_COLUMNS]
            feat_df = store.get_historical_features(
                entity_df=df, features=feature_refs
            ).to_df()
            df = df.merge(feat_df, on=["symbol", "event_time"], how="left")
            feature_names = list(FEATURE_COLUMNS)
        else:
            df, feature_names, _, _ = _extract_features(
                df,
                feature_names,
                n_jobs=n_jobs,
                model_json=model_json,
                news_embeddings=news_embeddings_df,
                news_embedding_window=news_window_param,
                news_embedding_horizon=news_horizon_seconds,
                config=feature_config,
                **feature_extra_kwargs,
            )
        meta_entry = technical_features._FEATURE_METADATA.get("__news_embeddings__")
        if meta_entry is not None:
            seq_arr = np.array(meta_entry.get("sequences", []), dtype=float, copy=True)
            if seq_arr.size:
                news_seq_list.append(seq_arr)
                if news_meta is None:
                    news_meta = {
                        key: meta_entry.get(key)
                        for key in (
                            "window",
                            "dimension",
                            "columns",
                            "horizon_seconds",
                        )
                    }
        FeatureSchema.validate(df[feature_names], lazy=True)
        if HAS_DASK and isinstance(df, dd.DataFrame):
            df = df.compute()
            if "symbol" in df.columns:
                symbol_batches.append(df["symbol"].astype(str).to_numpy())
            X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy(dtype=float)
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                signs = np.unique(np.sign(profits[np.isfinite(profits)]))
                if signs.size <= 1 and label_col is not None:
                    y = df[label_col].to_numpy(dtype=float)
                else:
                    y = (profits > 0).astype(float)
                has_profit = True
                if "event_time" in df.columns:
                    event_times = pd.to_datetime(
                        df["event_time"], errors="coerce"
                    ).to_numpy()
                else:
                    event_times = np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy(dtype=float)
                profits = np.zeros_like(y)
                has_profit = False
                event_times = (
                    pd.to_datetime(df["event_time"], errors="coerce").to_numpy()
                    if "event_time" in df.columns
                    else np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                )
        elif isinstance(df, pd.DataFrame):
            if "symbol" in df.columns:
                symbol_batches.append(df["symbol"].astype(str).to_numpy())
            X = df[feature_names].fillna(0.0).to_numpy(dtype=float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy(dtype=float)
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                signs = np.unique(np.sign(profits[np.isfinite(profits)]))
                if signs.size <= 1 and label_col is not None:
                    y = df[label_col].to_numpy(dtype=float)
                else:
                    y = (profits > 0).astype(float)
                has_profit = True
                if "event_time" in df.columns:
                    event_times = pd.to_datetime(
                        df["event_time"], errors="coerce"
                    ).to_numpy()
                else:
                    event_times = np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                ret_cols = [
                    c for c in ["event_time", "symbol", "profit"] if c in df.columns
                ]
                returns_df = df[ret_cols] if ret_cols else None
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy(dtype=float)
                profits = np.zeros_like(y)
                has_profit = False
                event_times = (
                    pd.to_datetime(df["event_time"], errors="coerce").to_numpy()
                    if "event_time" in df.columns
                    else np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                )
                returns_df = None
        elif HAS_POLARS and isinstance(df, pl.DataFrame):
            if "symbol" in df.columns:
                symbol_batches.append(df["symbol"].to_pandas().astype(str).to_numpy())
            X = df.select(feature_names).fill_null(0.0).to_numpy().astype(float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy().astype(float)
                label_col = next(
                    (c for c in df.columns if str(c).startswith("label")), None
                )
                signs = np.unique(np.sign(profits[np.isfinite(profits)]))
                if signs.size <= 1 and label_col is not None:
                    y = df[label_col].to_numpy().astype(float)
                else:
                    y = (profits > 0).astype(float)
                has_profit = True
                if "event_time" in df.columns:
                    event_times = pd.to_datetime(
                        df["event_time"].to_numpy(), errors="coerce"
                    ).to_numpy()
                else:
                    event_times = np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                ret_cols = [
                    c for c in ["event_time", "symbol", "profit"] if c in df.columns
                ]
                returns_df = df[ret_cols].to_pandas() if ret_cols else None
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy().astype(float)
                profits = np.zeros_like(y)
                has_profit = False
                event_times = (
                    pd.to_datetime(
                        df["event_time"].to_numpy(), errors="coerce"
                    ).to_numpy()
                    if "event_time" in df.columns
                    else np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                )
                returns_df = None
        else:  # pragma: no cover - defensive
            raise TypeError("Unsupported DataFrame type")

        if symbol_batches:
            symbols = np.concatenate(symbol_batches)
        else:
            symbols = np.array([], dtype=str)

    if active_subset:
        X, feature_names = filter_feature_matrix(X, feature_names, active_subset)
        filtered_feature_subset = list(feature_names)

    if news_seq_list:
        news_sequences = np.concatenate(news_seq_list, axis=0)
    else:
        news_sequences = None

    if profile and feature_prof is not None:
        feature_prof.disable()
        feature_prof.dump_stats(str(profiles_dir / "feature_extraction.prof"))
    span_ctx.__exit__(None, None, None)

    if has_profit and profits.size:
        cost = fee_per_trade + np.abs(profits) * slippage_bps * 1e-4
        profits = profits - cost
        y = (profits > 0).astype(float)
    weight_times = event_times if event_times.size else None
    sample_weight = build_sample_weights(
        profits,
        weight_times,
        half_life_days=half_life_days,
        use_volatility=vol_weight,
    )
    autoencoder_info: dict[str, Any] | None = None
    if bool(kwargs.get("use_autoencoder")):
        original_features = list(feature_names)
        path_override = (
            kwargs.get("autoencoder_path")
            or kwargs.get("autoencoder_model")
            or kwargs.get("autoencoder_checkpoint")
        )
        if path_override:
            ae_path = Path(path_override)
            if not ae_path.is_absolute():
                ae_path = out_dir / ae_path
        else:
            ae_path = out_dir / "autoencoder.pt"
        latent_override = kwargs.get("autoencoder_dim")
        if latent_override is None:
            latent_value = min(8, X.shape[1]) if X.shape[1] else 1
        else:
            latent_value = int(latent_override)
        latent_value = int(max(1, min(latent_value, X.shape[1] or 1)))
        embeddings = encode_with_autoencoder(X, ae_path, latent_dim=latent_value)
        metadata = load_autoencoder_metadata(ae_path) or {}
        metadata = dict(metadata)
        metadata.setdefault("latent_dim", int(embeddings.shape[1]))
        metadata["feature_names"] = [f"ae_{i}" for i in range(embeddings.shape[1])]
        metadata["input_features"] = original_features
        try:
            rel_weights = os.path.relpath(ae_path, out_dir)
        except ValueError:
            rel_weights = str(ae_path)
        metadata["weights_file"] = rel_weights
        meta_path = autoencoder_metadata_path(ae_path)
        if meta_path.exists():
            try:
                rel_meta = os.path.relpath(meta_path, out_dir)
            except ValueError:
                rel_meta = str(meta_path)
            metadata["metadata_file"] = rel_meta
        save_autoencoder_metadata(ae_path, metadata)
        autoencoder_info = metadata
        X = embeddings
        feature_names = metadata["feature_names"]
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
        if sequence_model:
            logger.info(
                "Skipping curriculum learning for sequence model type %s",
                model_type,
            )
        else:
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

    if pretrain_mask is not None and HAS_TORCH:
        enc_path = Path(pretrain_mask)
        if enc_path.exists():
            state = torch.load(enc_path, map_location="cpu")
            arch = state.get("architecture", [])
            if arch:
                encoder_inputs = list(feature_names)
                encoder = torch.nn.Linear(int(arch[0]), int(arch[1]))
                encoder.load_state_dict(state["state_dict"])
                encoder.eval()
                with torch.no_grad():
                    X = encoder(torch.as_tensor(X, dtype=torch.float32)).numpy()
                feature_names = [f"enc_{i}" for i in range(X.shape[1])]
                state_dict = state.get("state_dict", {})
                weight_tensor = state_dict.get("weight")
                bias_tensor = state_dict.get("bias")
                if isinstance(weight_tensor, torch.Tensor):
                    weight_values = weight_tensor.detach().cpu().numpy()
                elif weight_tensor is not None:
                    weight_values = np.asarray(weight_tensor, dtype=float)
                else:
                    weight_values = encoder.weight.detach().cpu().numpy()
                if isinstance(bias_tensor, torch.Tensor):
                    bias_values: np.ndarray | None = bias_tensor.detach().cpu().numpy()
                elif bias_tensor is not None:
                    bias_values = np.asarray(bias_tensor, dtype=float)
                else:
                    bias_attr = getattr(encoder, "bias", None)
                    bias_values = (
                        bias_attr.detach().cpu().numpy()
                        if isinstance(bias_attr, torch.Tensor)
                        else None
                    )
                dest_path = out_dir / enc_path.name
                try:
                    if enc_path.resolve() != dest_path.resolve():
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(enc_path, dest_path)
                except OSError:
                    dest_path = enc_path
                try:
                    rel_weights = os.path.relpath(dest_path, out_dir)
                except ValueError:
                    rel_weights = str(dest_path)
                encoder_meta = {
                    "architecture": [int(arch[0]), int(arch[1])],
                    "input_dim": int(arch[0]) if arch else len(encoder_inputs),
                    "latent_dim": int(arch[1]) if len(arch) > 1 else int(X.shape[1]),
                    "mask_ratio": float(state.get("mask_ratio", 0.0)),
                    "input_features": encoder_inputs,
                    "weights_file": rel_weights,
                    "weights": np.asarray(weight_values, dtype=float).tolist(),
                    "bias": (
                        np.asarray(bias_values, dtype=float).tolist()
                        if bias_values is not None
                        else None
                    ),
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
        if not np.isfinite(condensed).all():
            logger.debug("Skipping correlation clustering due to non-finite distances")
        else:
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

    window_length = max(1, int(kwargs.get("window", 1)))
    if model_type == "crossmodal":
        if news_sequences is None:
            raise ValueError(
                "news embeddings must be provided when using model_type='crossmodal'"
            )
        if news_sequences.shape[0] != X.shape[0]:
            raise ValueError("news embeddings are not aligned with feature rows")
    if sequence_model:
        (
            sequence_data,
            R,
            X,
            y,
            profits,
            sample_weight,
            returns_df,
            news_sequence_data,
            symbols_out,
        ) = build_window_sequences(
            X,
            y,
            profits,
            sample_weight,
            window_length=window_length,
            returns_df=returns_df,
            news_sequences=news_sequences,
            symbols=symbols,
            regime_features=R,
        )
        if symbols_out is not None:
            symbols = symbols_out
        else:
            symbols = np.array([], dtype=str)
    else:
        sequence_data = None
        news_sequence_data = None

    sample_weight = normalise_weights(sample_weight)
    weight_stats = summarise_weights(sample_weight)
    if sample_weight.size:
        logger.info("Sample weight stats: %s", weight_stats)
        metric_payload = {
            "model_type": model_type,
            "half_life_days": float(half_life_days),
            "vol_weight": bool(vol_weight),
            **weight_stats,
        }
        try:
            add_metric("train_sample_weights", metric_payload)
        except Exception:  # pragma: no cover - metrics logging best effort
            logger.exception("Failed to record sample weight metrics")
    else:
        weight_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    symbol_indices = np.array([], dtype=int)
    symbol_names_ctx: list[str] = []
    symbol_embeddings_ctx = np.zeros((0, 0), dtype=float)
    neighbor_lists_ctx: list[list[int]] = []
    neighbor_order_ctx: dict[str, list[str]] = {}
    if model_type == "multi_symbol":
        symbol_map, symbol_names_ctx, symbol_embeddings_ctx, neighbor_lists_ctx, neighbor_order_ctx = prepare_symbol_context(
            symbols.tolist(), kwargs.get("symbol_graph")
        )
        if symbols.size:
            symbol_indices = np.array([symbol_map.get(str(sym), -1) for sym in symbols], dtype=int)
            if np.any(symbol_indices < 0):
                raise ValueError("Encountered symbols without graph embeddings")
        else:
            symbol_indices = np.array([], dtype=int)

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
    if mlflow_active and not HAS_MLFLOW:
        raise TrainingPipelineError("mlflow is required for tracking")
    if mlflow_active:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

    span_model = tracer.start_as_current_span("model_fit")
    if profile and fit_prof is not None:
        fit_prof.enable()
    span_model.__enter__()
    mlflow_run_info: dict[str, str] = {}
    run_ctx = mlflow.start_run() if mlflow_active else nullcontext()
    with run_ctx:
        if mlflow_active:
            current_run = mlflow.active_run()
            if current_run is not None:
                mlflow_run_info = {
                    "run_id": str(current_run.info.run_id),
                    "experiment_id": str(current_run.info.experiment_id),
                }
        n_splits = int(kwargs.get("n_splits", 3))
        gap = int(kwargs.get("cv_gap", 1))
        splitter = PurgedWalkForward(n_splits=n_splits, gap=gap)
        try:
            splits = list(splitter.split(X))
        except ValueError:
            logger.debug(
                "Insufficient samples for %d folds; falling back to hold-out split",
                n_splits,
            )
            if X.shape[0] <= 1:
                splits = [(np.array([0]), np.array([0]))]
            else:
                train_idx = np.arange(0, X.shape[0] - 1)
                val_idx = np.array([X.shape[0] - 1])
                splits = [(train_idx, val_idx)]
        if model_type == "crossmodal":
            if sequence_data is None or news_sequence_data is None:
                raise ValueError("crossmodal model requires sequence inputs")
            model_inputs = (sequence_data, news_sequence_data)
        elif model_type == "multi_symbol":
            model_inputs = (X, symbol_indices)
        else:
            model_inputs = sequence_data if sequence_model else X
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
        metrics: dict[str, object] = {}

        def _evaluate_thresholds_for_fold(
            fold_preds: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
        ) -> tuple[float, dict[str, object], list[tuple[int, dict[str, object]]]]:
            if not fold_preds:
                return 0.5, {}, []
            fold_preds = sorted(fold_preds, key=lambda x: x[0])
            y_all = np.concatenate([fp[1] for fp in fold_preds])
            prob_all = np.concatenate([fp[2] for fp in fold_preds])
            profit_all = np.concatenate([fp[3] for fp in fold_preds])
            try:
                best_threshold, best_metrics = search_decision_threshold(
                    y_all,
                    prob_all,
                    profit_all,
                    objective=threshold_objective,
                    threshold_grid=threshold_grid_values,
                    metric_names=metric_names,
                    max_drawdown=max_drawdown,
                    var_limit=var_limit,
                )
            except ValueError:
                return 0.5, {}, []
            fold_metrics: list[tuple[int, dict[str, object]]] = []
            for fold_idx, y_val, prob_val, prof_val in fold_preds:
                returns_fold = prof_val * (prob_val >= best_threshold)
                metrics_fold = _classification_metrics_cached(
                    y_val,
                    prob_val,
                    returns_fold,
                    selected=metric_names,
                    threshold=best_threshold,
                )
                metrics_fold.setdefault("max_drawdown", eval_max_drawdown(returns_fold))
                metrics_fold.setdefault("var_95", eval_var_95(returns_fold))
                metrics_fold["threshold"] = float(best_threshold)
                metrics_fold["threshold_objective"] = threshold_objective
                fold_metrics.append((fold_idx, metrics_fold))
            return best_threshold, best_metrics, fold_metrics

        if distributed and not HAS_RAY:
            raise RuntimeError("ray is required for distributed execution")
        for params in param_grid:
            fold_predictions: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            if distributed and HAS_RAY:
                if model_type == "crossmodal":
                    price_ref = ray.put(sequence_data)
                    news_ref = ray.put(news_sequence_data)
                else:
                    X_ref = ray.put(model_inputs)
                y_ref = ray.put(y)
                profits_ref = ray.put(profits)
                weight_ref = ray.put(sample_weight)
                R_ref = ray.put(R) if model_type == "moe" else None

                @ray.remote
                def _run_fold(tr_idx, val_idx, fold):
                    if model_type == "crossmodal":
                        price_data = ray.get(price_ref)
                        news_data = ray.get(news_ref)
                    elif model_type == "multi_symbol":
                        data_feat, data_sym = ray.get(X_ref)
                    else:
                        data = ray.get(X_ref)
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
                        builder_kwargs = dict(
                            **gpu_kwargs, **params, **extra_model_params
                        )
                        builder_kwargs["sample_weight"] = weights[tr_idx]
                        if model_type in {"moe", "multi_symbol"} or sequence_model:
                            builder_kwargs["grad_clip"] = grad_clip
                        if meta_init is not None:
                            builder_kwargs["init_weights"] = meta_init
                        if model_type == "multi_symbol":
                            builder_kwargs.update(
                                {
                                    "symbol_names": symbol_names_ctx,
                                    "embeddings": symbol_embeddings_ctx,
                                    "neighbor_index": neighbor_lists_ctx,
                                }
                            )
                        if model_type == "crossmodal":
                            train_input = (
                                price_data[tr_idx],
                                news_data[tr_idx],
                            )
                            model_fold, pred_fn = builder(
                                train_input,
                                y[tr_idx],
                                **builder_kwargs,
                            )
                            prob_val = pred_fn(
                                (
                                    price_data[val_idx],
                                    news_data[val_idx],
                                )
                            )
                        elif model_type == "multi_symbol":
                            local_kwargs = dict(builder_kwargs)
                            local_kwargs["symbol_ids"] = data_sym[tr_idx]
                            model_fold, pred_fn = builder(
                                data_feat[tr_idx],
                                y[tr_idx],
                                **local_kwargs,
                            )
                            prob_val = pred_fn((data_feat[val_idx], data_sym[val_idx]))
                        else:
                            model_fold, pred_fn = builder(
                                data[tr_idx],
                                y[tr_idx],
                                **builder_kwargs,
                            )
                            prob_val = pred_fn(data[val_idx])
                    profits_val = profits[val_idx]
                    y_val = y[val_idx]
                    return (
                        fold,
                        np.asarray(y_val, dtype=float),
                        np.asarray(prob_val, dtype=float),
                        np.asarray(profits_val, dtype=float),
                    )

                futures = [
                    _run_fold.remote(tr_idx, val_idx, fold)
                    for fold, (tr_idx, val_idx) in enumerate(splits)
                    if len(np.unique(y[tr_idx])) >= 2
                ]
                results = ray.get(futures)
                for fold, y_val, prob_val, prof_val in results:
                    fold_predictions.append((fold, y_val, prob_val, prof_val))
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
                    builder_kwargs = dict(
                        **gpu_kwargs, **params, **extra_model_params
                    )
                    builder_kwargs["sample_weight"] = sample_weight[tr_idx]
                    if model_type in {"moe", "multi_symbol"} or sequence_model:
                        builder_kwargs["grad_clip"] = grad_clip
                    if meta_init is not None:
                        builder_kwargs["init_weights"] = meta_init
                    if model_type == "multi_symbol":
                        builder_kwargs.update(
                            {
                                "symbol_names": symbol_names_ctx,
                                "embeddings": symbol_embeddings_ctx,
                                "neighbor_index": neighbor_lists_ctx,
                            }
                        )
                    if model_type == "crossmodal":
                        price_train = sequence_data[tr_idx]
                        news_train = news_sequence_data[tr_idx]
                        model_fold, pred_fn = builder(
                            (price_train, news_train),
                            y[tr_idx],
                            **builder_kwargs,
                        )
                        prob_val = pred_fn(
                            (
                                sequence_data[val_idx],
                                news_sequence_data[val_idx],
                            )
                        )
                    elif model_type == "multi_symbol":
                        data_feat, data_sym = model_inputs
                        local_kwargs = dict(builder_kwargs)
                        local_kwargs["symbol_ids"] = data_sym[tr_idx]
                        model_fold, pred_fn = builder(
                            data_feat[tr_idx],
                            y[tr_idx],
                            **local_kwargs,
                        )
                        prob_val = pred_fn((data_feat[val_idx], data_sym[val_idx]))
                    else:
                        model_fold, pred_fn = builder(
                            model_inputs[tr_idx],
                            y[tr_idx],
                            **builder_kwargs,
                        )
                        prob_val = pred_fn(model_inputs[val_idx])
                    profits_val = profits[val_idx]
                    if profile and eval_prof is not None:
                        eval_prof.enable()
                    y_val = y[val_idx]
                    prob_sel = np.asarray(prob_val, dtype=float)
                    prof_sel = np.asarray(profits_val, dtype=float)
                    y_sel = np.asarray(y_val, dtype=float)
                    if profile and eval_prof is not None:
                        eval_prof.disable()
                    fold_predictions.append((fold, y_sel, prob_sel, prof_sel))
            (
                threshold_value,
                combined_metrics,
                fold_metric_entries,
            ) = _evaluate_thresholds_for_fold(fold_predictions)
            if not fold_metric_entries:
                continue
            fold_metrics = [metrics_dict for _, metrics_dict in fold_metric_entries]
            for fold_idx, metrics_dict in fold_metric_entries:
                logger.info(
                    "Fold %d params %s metrics %s", fold_idx + 1, params, metrics_dict
                )
                if mlflow_active:
                    for k, v in metrics_dict.items():
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            mlflow.log_metric(f"fold{fold_idx + 1}_{k}", float(v))
        if not metrics:
            logger.warning(
                "No decision threshold satisfied risk constraints; using fallback"
            )
            if fold_metrics:
                metrics = fold_metrics[0].copy()
            else:
                metrics = {
                    "accuracy": float(np.mean(y)),
                    "profit": float(np.sum(profits)),
                }
            threshold_value = threshold_value or 0.5
            combined_metrics = combined_metrics or {}
        agg = {
            k: float(
                np.nanmean(
                    [m[k] for m in fold_metrics if isinstance(m.get(k), (int, float))]
                )
            )
            for k in fold_metrics[0].keys()
            if k != "reliability_curve"
        }
        agg["threshold"] = float(threshold_value)
        agg["threshold_objective"] = threshold_objective
        for key in [
            "reliability_curve",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece",
        ]:
            if key in combined_metrics and combined_metrics[key] is not None:
                agg[key] = combined_metrics[key]
        skip_due_to_drawdown = (
            max_drawdown is not None and agg.get("max_drawdown", 0.0) > max_drawdown
        )
        if skip_due_to_drawdown:
            logger.info(
                "Skipping params %s due to max drawdown %.6f exceeding limit %.6f",
                params,
                agg.get("max_drawdown", 0.0),
                max_drawdown,
            )
        skip_due_to_var = var_limit is not None and agg.get("var_95", 0.0) > var_limit
        if skip_due_to_var:
            logger.info(
                "Skipping params %s due to var_95 %.6f exceeding limit %.6f",
                params,
                agg.get("var_95", 0.0),
                var_limit,
            )
        if not (skip_due_to_drawdown or skip_due_to_var):
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
                metrics = agg.copy()
                metrics.setdefault("threshold", float(threshold_value))
                metrics.setdefault("threshold_objective", threshold_objective)
        selected_threshold = float(metrics.get("threshold", 0.5))
        if profile and eval_prof is not None:
            eval_prof.dump_stats(str(profiles_dir / "evaluation.prof"))
        metrics.setdefault("threshold", selected_threshold)
        metrics.setdefault("threshold_objective", threshold_objective)
        if max_drawdown is not None and metrics.get("max_drawdown", 0.0) > max_drawdown:
            raise ValueError("Selected model exceeds max_drawdown limit")
        if var_limit is not None and metrics.get("var_95", 0.0) > var_limit:
            raise ValueError("Selected model exceeds var_95 limit")
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
            builder_kwargs = dict(
                **gpu_kwargs, **(best_params or {}), **extra_model_params
            )
            builder_kwargs["sample_weight"] = sample_weight
            if model_type in {"moe", "multi_symbol"} or sequence_model:
                builder_kwargs["grad_clip"] = grad_clip
            if meta_init is not None:
                builder_kwargs["init_weights"] = meta_init
            if model_type == "multi_symbol":
                builder_kwargs.update(
                    {
                        "symbol_names": symbol_names_ctx,
                        "embeddings": symbol_embeddings_ctx,
                        "neighbor_index": neighbor_lists_ctx,
                        "symbol_ids": symbol_indices,
                    }
                )
                data_features, _symbol_idx = model_inputs
                model_data, predict_fn = builder(
                    data_features,
                    y,
                    **builder_kwargs,
                )
            else:
                model_data, predict_fn = builder(
                    model_inputs,
                    y,
                    **builder_kwargs,
                )

        # --- SHAP based feature selection ---------------------------------
        shap_threshold = float(kwargs.get("shap_threshold", 0.0))
        if not sequence_model:
            mean_abs: np.ndarray | None = None
            try:
                mean_abs = mean_absolute_shap(
                    predict_fn, X, model_type=model_type
                )
            except ImportError:  # pragma: no cover - optional dependency missing
                mean_abs = None
            except Exception:  # pragma: no cover - shap is optional
                logger.exception("Failed to compute SHAP values")
                mean_abs = None

            if (mean_abs is None or not mean_abs.size) and model_type == "logreg":
                coef = np.asarray(model_data.get("coefficients", []), dtype=float)
                if coef.size == X.shape[1]:
                    mean_abs = np.abs(coef) * np.std(X, axis=0)

            if mean_abs is not None and mean_abs.size:
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
                        if model_type == "multi_symbol":
                            model_inputs = (X, symbol_indices)
                        else:
                            model_inputs = X
                        builder_kwargs = dict(
                            **gpu_kwargs,
                            **(best_params or {}),
                            **extra_model_params,
                        )
                        builder_kwargs["sample_weight"] = sample_weight
                        if model_type in {"moe", "multi_symbol"} or sequence_model:
                            builder_kwargs["grad_clip"] = grad_clip
                        if model_type == "multi_symbol":
                            builder_kwargs.update(
                                {
                                    "symbol_names": symbol_names_ctx,
                                    "embeddings": symbol_embeddings_ctx,
                                    "neighbor_index": neighbor_lists_ctx,
                                    "symbol_ids": symbol_indices,
                                }
                            )
                            model_data, predict_fn = builder(
                                model_inputs[0],
                                y,
                                **builder_kwargs,
                            )
                        else:
                            model_data, predict_fn = builder(
                                model_inputs,
                                y,
                                **builder_kwargs,
                            )
        model_obj = getattr(predict_fn, "model", None)

        # --- Probability calibration ---------------------------------------
        calibration_info: dict[str, object] | None = None
        if model_type != "moe" and model_type != "multi_symbol" and not sequence_model:
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
            data_eval = model_inputs
            probas = predict_fn(data_eval)
        preds = (probas >= 0.5).astype(float)
        score = float((preds == y).mean())
        meta_lookup = getattr(technical_features, "_FEATURE_METADATA", {})
        feature_metadata: list[FeatureMetadata] = []
        for fn in feature_names:
            meta_entry = meta_lookup.get(fn)
            if isinstance(meta_entry, FeatureMetadata):
                feature_metadata.append(meta_entry)
                continue
            if isinstance(meta_entry, dict):
                original = str(meta_entry.get("original_column", fn))
                transformations = list(meta_entry.get("transformations", []))
                parameters = dict(meta_entry.get("parameters", {}))
                feature_metadata.append(
                    FeatureMetadata(
                        original_column=original,
                        transformations=transformations,
                        parameters=parameters,
                    )
                )
            else:
                feature_metadata.append(FeatureMetadata(original_column=fn))
        model = {
            "feature_names": feature_names,
            "feature_metadata": feature_metadata,
            **model_data,
            "model_type": model_type,
        }
        sent_embed_meta = technical_features._FEATURE_METADATA.get(
            "sentiment_embeddings"
        )
        if isinstance(sent_embed_meta, dict):
            sent_cols = list(sent_embed_meta.get("columns", []))
            if sent_cols:
                model["sentiment_feature"] = sent_cols[0]
                model["sentiment_embeddings"] = {
                    "columns": sent_cols,
                    "dimension": int(
                        sent_embed_meta.get("dimension", len(sent_cols))
                    ),
                    "source": sent_embed_meta.get("source", "news_sentiment"),
                }
        if model_type == "multi_symbol":
            if "neighbor_order" not in model:
                model["neighbor_order"] = neighbor_order_ctx
            if "attention_weights" not in model:
                model["attention_weights"] = {}
        if model_type == "crossmodal" and news_meta is not None:
            model["news_embeddings"] = {
                "window": int(news_meta.get("window", news_window_param)),
                "dimension": int(
                    news_meta.get(
                        "dimension",
                        (news_sequence_data.shape[-1]
                         if news_sequence_data is not None
                         else 0),
                    )
                ),
                "columns": list(news_meta.get("columns", [])),
                "horizon_seconds": float(
                    news_meta.get("horizon_seconds", news_horizon_seconds)
                ),
            }
        if half_life_days > 0.0:
            model["half_life_days"] = float(half_life_days)
        if vol_weight:
            model["volatility_weighting"] = True
        if sample_weight.size:
            model["sample_weight_stats"] = weight_stats
        if meta_init is not None:
            meta_entry = {"weights": meta_init.tolist()}
            if meta_info:
                meta_entry.update(meta_info)
            meta_entry["adapted"] = True
            model.setdefault("meta", meta_entry)
            model.setdefault("meta_weights", meta_init.tolist())
        if autoencoder_info is not None:
            auto_meta_copy = json.loads(json.dumps(autoencoder_info))
            model["autoencoder"] = auto_meta_copy
        if encoder_meta is not None:
            model["masked_encoder"] = encoder_meta
        if getattr(technical_features, "_DEPTH_CNN_STATE", None) is not None:
            model["depth_cnn"] = technical_features._DEPTH_CNN_STATE
        if getattr(technical_features, "_CSD_PARAMS", None) is not None:
            model["csd_params"] = technical_features._CSD_PARAMS
        if getattr(technical_features, "_GRAPH_SNAPSHOT", None) is not None:
            model["graph_snapshot"] = technical_features._GRAPH_SNAPSHOT
        if getattr(technical_features, "_GNN_STATE", None) is not None:
            model["gnn_state"] = technical_features._GNN_STATE
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
        if model_json and Path(model_json).exists():
            try:
                existing = json.loads(Path(model_json).read_text())
                sym = existing.get("symbolic_indicators")
                if sym:
                    model["symbolic_indicators"] = sym
            except Exception:
                pass
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
        serialised_cv_metrics = serialise_metric_values(metrics)
        session_cv_metrics = serialise_metric_values(metrics)
        serialised_fold_metrics = [
            serialise_metric_values(m) for m in best_fold_metrics
        ]
        if "session_models" not in model:
            sm_keys = [
                "coefficients",
                "intercept",
                "feature_mean",
                "feature_std",
                "clip_low",
                "clip_high",
            ]
            if autoencoder_info is not None:
                sm_keys.append("autoencoder")
            if encoder_meta is not None:
                sm_keys.append("masked_encoder")
            model["session_models"] = {
                "asian": {k: model[k] for k in sm_keys if k in model}
            }
        else:
            if autoencoder_info is not None:
                for sess in model["session_models"].values():
                    sess.setdefault(
                        "autoencoder", json.loads(json.dumps(autoencoder_info))
                    )
            if encoder_meta is not None:
                for sess in model["session_models"].values():
                    sess.setdefault("masked_encoder", json.loads(json.dumps(encoder_meta)))
        model["cv_metrics"] = serialised_cv_metrics
        model["threshold"] = float(selected_threshold)
        model["decision_threshold"] = float(selected_threshold)
        model["threshold_objective"] = threshold_objective
        ensemble_cfg = model.get("ensemble")
        if isinstance(ensemble_cfg, dict):
            ensemble_cfg.setdefault("threshold", float(selected_threshold))
        model["cv_accuracy"] = metrics.get("accuracy", 0.0)
        model["cv_profit"] = metrics.get("profit", 0.0)
        model["conformal_lower"] = 0.0
        model["conformal_upper"] = 1.0
        model["session_models"]["asian"]["cv_metrics"] = serialised_fold_metrics
        model["session_models"]["asian"]["threshold"] = float(selected_threshold)
        model["session_models"]["asian"]["metrics"] = session_cv_metrics
        model["session_models"]["asian"]["threshold_objective"] = threshold_objective
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
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        }
        end_time = datetime.now(UTC)
        duration = max(0.0, (end_time - start_time).total_seconds())
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = model.setdefault("metadata", {})
        metadata["seed"] = random_seed
        metadata["training_started_at"] = start_time.isoformat()
        metadata["training_completed_at"] = end_time.isoformat()
        metadata["training_duration_seconds"] = duration
        n_samples = int(getattr(X, "shape", (0,))[0]) if hasattr(X, "shape") else 0
        if hasattr(X, "ndim") and X.ndim >= 2:
            n_features_val = int(X.shape[1])
        else:
            n_features_val = int(len(feature_names))
        metadata["n_samples"] = n_samples
        metadata["n_features"] = n_features_val
        if filtered_feature_subset:
            metadata["selected_features"] = list(filtered_feature_subset)
        if model_type == "crossmodal" and news_meta is not None and news_sequence_data is not None:
            metadata["news_embeddings"] = {
                "window": int(news_meta.get("window", news_window_param)),
                "dimension": int(news_sequence_data.shape[-1]),
                "columns": list(news_meta.get("columns", [])),
                "horizon_seconds": float(
                    news_meta.get("horizon_seconds", news_horizon_seconds)
                ),
            }
        env_info = metadata.setdefault("environment", {})
        env_info["python"] = sys.version.split()[0]
        env_info["platform"] = platform.platform()
        experiment_meta = metadata.get("experiment", {})
        if not experiment_meta:
            experiment_meta = {"run_id": run_identifier, "tracking": "offline"}
        else:
            experiment_meta.setdefault("run_id", run_identifier)
            experiment_meta.setdefault("tracking", "offline")
        if mlflow_active:
            experiment_meta["tracking"] = "mlflow"
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri:
                experiment_meta["tracking_uri"] = tracking_uri
            if experiment_name:
                experiment_meta["experiment_name"] = experiment_name
            if mlflow_run_info:
                experiment_meta["mlflow_run_id"] = mlflow_run_info.get("run_id")
                experiment_meta["mlflow_experiment_id"] = mlflow_run_info.get(
                    "experiment_id"
                )
        if dvc_repo_path is not None:
            experiment_meta["dvc_repo"] = str(dvc_repo_path)
        metadata["experiment"] = experiment_meta
        if config_hash:
            model["config_hash"] = config_hash
            metadata["config_hash"] = config_hash
        if config_snapshot:
            normalised = json.loads(json.dumps(config_snapshot, default=str))
            config_snapshot_path = out_dir / "config_snapshot.json"
            config_snapshot_path.write_text(
                json.dumps(normalised, indent=2, sort_keys=True)
            )
            snapshot_digest = hashlib.sha256(
                json.dumps(normalised, sort_keys=True).encode("utf-8")
            ).hexdigest()
            metadata["config_snapshot"] = normalised
            try:
                metadata["config_snapshot_path"] = str(
                    config_snapshot_path.relative_to(out_dir)
                )
            except ValueError:
                metadata["config_snapshot_path"] = str(config_snapshot_path)
            metadata["config_snapshot_hash"] = snapshot_digest
            if not config_hash:
                metadata.setdefault("config_hash", snapshot_digest)
                model.setdefault("config_hash", snapshot_digest)
        metadata["data_hashes_path"] = "data_hashes.json"
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
                    weights, link = _hrp_cached(pivot)
                    model["hrp_weights"] = weights.to_dict()
                    model["hrp_dendrogram"] = link.tolist()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to compute HRP allocation")
        deps_file = write_dependency_snapshot(out_dir)
        try:
            relative_deps = deps_file.relative_to(out_dir)
        except ValueError:
            relative_deps = deps_file
        metadata["dependencies_file"] = str(relative_deps)
        metadata["dependencies_hash"] = hashlib.sha256(
            deps_file.read_bytes()
        ).hexdigest()
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
        numeric_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating)) and not np.isnan(float(v))
        }
        generate_model_card(
            model_params,
            numeric_metrics,
            out_dir / "model_card.md",
            dependencies_path=deps_file,
        )
        if model_obj is not None:
            try:
                from botcopier.onnx_utils import export_model

                export_model(model_obj, X, out_dir / "model.onnx")
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to export ONNX model")
        artifacts: list[Path] = [
            data_dir,
            out_dir / "model.json",
            out_dir / "data_hashes.json",
            deps_file,
        ]
        if config_snapshot_path is not None:
            artifacts.append(config_snapshot_path)
        version_artifacts_with_dvc(dvc_repo_path, artifacts)
        if mlflow_active:
            base_params: dict[str, object] = {
                "model_type": model_type,
                "n_features": len(feature_names),
                "random_seed": random_seed,
                "grad_clip": grad_clip,
                "fee_per_trade": fee_per_trade,
                "slippage_bps": slippage_bps,
                "distributed": distributed,
                "use_gpu": use_gpu,
                "hrp_allocation": hrp_allocation,
                "strategy_search": strategy_search,
            }
            if n_jobs is not None:
                base_params["n_jobs"] = n_jobs
            if features is not None:
                base_params["requested_features"] = list(features)
            if filtered_feature_subset:
                base_params["selected_features"] = list(filtered_feature_subset)
            if regime_features is not None:
                base_params["regime_features"] = list(regime_features)
            if dvc_repo_path is not None:
                base_params["dvc_repo"] = dvc_repo_path
            mlflow.log_params(
                {k: serialize_mlflow_param(v) for k, v in base_params.items()}
            )
            if best_params:
                mlflow.log_params(
                    {
                        f"hp_{k}": serialize_mlflow_param(v)
                        for k, v in best_params.items()
                    }
                )
            mlflow.log_metric("train_accuracy", float(score))
            aggregated_metrics = {
                f"cv_{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            }
            if aggregated_metrics:
                mlflow.log_metrics(aggregated_metrics)
            mlflow.log_artifact(str(out_dir / "model.json"), artifact_path="model")
            mlflow.log_artifact(
                str(out_dir / "data_hashes.json"), artifact_path="model"
            )
            if config_snapshot_path is not None and config_snapshot_path.exists():
                mlflow.log_artifact(str(config_snapshot_path), artifact_path="model")
            model_uri = mlflow.get_artifact_uri("model/model.json")
            data_hash_uri = mlflow.get_artifact_uri("model/data_hashes.json")
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("data_hashes_uri", data_hash_uri)
            if config_snapshot_path is not None:
                snapshot_uri = mlflow.get_artifact_uri("model/config_snapshot.json")
                mlflow.log_param("config_snapshot_uri", snapshot_uri)
        span_eval.__exit__(None, None, None)
        return model_obj


def predict_expected_value(model: dict, X: np.ndarray) -> np.ndarray:
    """Compute expected profit predictions for feature matrix ``X``."""

    if "session_models" in model and model["session_models"]:
        params = next(iter(model["session_models"].values()))
    else:
        params = model

    features = np.asarray(X, dtype=float)
    feature_names = params.get("feature_names") or model.get("feature_names", [])
    autoencoder_meta = params.get("autoencoder") or model.get("autoencoder")
    metadata: Mapping[str, Any] | None = None
    schema_feature_names: Sequence[str] | None = None
    if autoencoder_meta:
        metadata = dict(autoencoder_meta)
        input_feature_names = metadata.get("input_features")
        if input_feature_names:
            schema_feature_names = list(input_feature_names)
        elif feature_names:
            schema_feature_names = list(feature_names)
    elif feature_names:
        schema_feature_names = list(feature_names)

    if schema_feature_names:
        if len(schema_feature_names) != features.shape[1]:
            raise ValueError(
                "feature matrix has %s columns but %s feature names provided"
                % (features.shape[1], len(schema_feature_names))
            )
        df = pd.DataFrame(features, columns=schema_feature_names)
        FeatureSchema.validate(df, lazy=True)

    if metadata is not None:
        features = apply_autoencoder_from_metadata(features, metadata)
        latent_names = metadata.get("feature_names")
        if latent_names:
            if len(latent_names) != features.shape[1]:
                raise ValueError(
                    "autoencoder produced %s features but %s names provided"
                    % (features.shape[1], len(latent_names))
                )
            feature_names = list(latent_names)
    clip_low = np.asarray(
        params.get("clip_low", model.get("clip_low", [])), dtype=float
    )
    clip_high = np.asarray(
        params.get("clip_high", model.get("clip_high", [])), dtype=float
    )
    if clip_low.size and clip_high.size and clip_low.shape == clip_high.shape:
        features = np.clip(features, clip_low, clip_high)

    mean = np.asarray(
        params.get("feature_mean", model.get("feature_mean", [])), dtype=float
    )
    std = np.asarray(
        params.get("feature_std", model.get("feature_std", [])), dtype=float
    )
    if mean.size and std.size and mean.shape == std.shape:
        denom = np.where(std == 0, 1.0, std)
        features = features - mean
        features = features / denom

    coef = np.asarray(params.get("coefficients", []), dtype=float)
    intercept = float(params.get("intercept", 0.0))
    logits = features @ coef + intercept
    prob = 1.0 / (1.0 + np.exp(-logits))

    pnl_model = params.get("pnl_model")
    if pnl_model:
        pnl_coef = np.asarray(pnl_model.get("coefficients", []), dtype=float)
        pnl_intercept = float(pnl_model.get("intercept", 0.0))
        pnl = features @ pnl_coef + pnl_intercept
    else:
        pnl = np.ones_like(prob)

    return prob * pnl


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
    if HAS_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
        has_gpu = True
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model_type = "logreg"
    if has_gpu and gpu_mem_gb >= 8.0:
        model_type = "tabtransformer"
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
        "--half-life",
        dest="half_life_days",
        type=float,
        default=0.0,
        help="half-life in days for exponential time-decay weighting",
    )
    p.add_argument(
        "--vol-weight",
        dest="vol_weight",
        action="store_true",
        help="scale sample weights by rolling profit volatility",
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
        "--controller-max-subset",
        dest="controller_max_subset_size",
        type=int,
        default=None,
        help="Maximum number of features per subset evaluated by the controller",
    )
    p.add_argument(
        "--controller-sample-size",
        dest="controller_episode_sample_size",
        type=int,
        default=None,
        help="Number of feature subsets sampled per controller episode",
    )
    p.add_argument(
        "--controller-baseline-momentum",
        dest="controller_baseline_momentum",
        type=float,
        default=None,
        help="Momentum for the controller reward baseline (0 disables)",
    )
    p.add_argument(
        "--use-meta",
        type=Path,
        dest="use_meta",
        help="Path to model.json with meta-weights",
    )
    args = p.parse_args()
    data_cfg, train_cfg, exec_cfg = load_settings(vars(args))
    setup_logging(enable_tracing=exec_cfg.trace, exporter=exec_cfg.trace_exporter)
    set_seed(train_cfg.random_seed)
    train(
        data_cfg.data_dir,
        data_cfg.out_dir,
        model_type=train_cfg.model_type,
        tracking_uri=train_cfg.tracking_uri,
        experiment_name=train_cfg.experiment_name,
        use_gpu=exec_cfg.use_gpu,
        random_seed=train_cfg.random_seed,
        metrics=train_cfg.metrics or args.metrics,
        grad_clip=train_cfg.grad_clip,
        half_life_days=train_cfg.half_life_days,
        vol_weight=train_cfg.vol_weight,
        profile=exec_cfg.profile,
        strategy_search=train_cfg.strategy_search,
        reuse_controller=train_cfg.reuse_controller,
        controller_max_subset_size=train_cfg.controller_max_subset_size,
        controller_episode_sample_size=train_cfg.controller_episode_sample_size,
        controller_baseline_momentum=train_cfg.controller_baseline_momentum,
        meta_weights=train_cfg.meta_weights,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
