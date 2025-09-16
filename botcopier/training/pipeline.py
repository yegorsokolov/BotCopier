"""Training pipeline orchestrating model training and evaluation."""

from __future__ import annotations

import argparse
import cProfile
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable, Sequence

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
from joblib import Memory
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

try:  # optional optuna support
    import optuna

    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover - optional
    optuna = None  # type: ignore
    _HAS_OPTUNA = False

try:  # optional numba support
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional

    def njit(*a, **k):  # pragma: no cover - simple stub
        def _inner(f):
            return f

        return _inner

    _HAS_NUMBA = False
from opentelemetry import trace
from pydantic import ValidationError

import botcopier.features.technical as technical_features
from automl.controller import AutoMLController
from botcopier.config.settings import (
    DataConfig,
    ExecutionConfig,
    TrainingConfig,
    load_settings,
)
from botcopier.exceptions import TrainingPipelineError
from botcopier.data.feature_schema import FeatureSchema
from botcopier.data.loading import _load_logs
from botcopier.features.anomaly import _clip_train_features
from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    _neutralize_against_market_index,
    configure_cache,
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
from metrics.aggregator import add_metric

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

if _HAS_TORCH:
    from botcopier.models.deep import TabTransformer, TCNClassifier
else:  # pragma: no cover - optional dependency
    TabTransformer = None  # type: ignore
    TCNClassifier = None  # type: ignore

try:  # optional mlflow dependency
    import mlflow  # type: ignore

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

try:  # optional dvc dependency
    from dvc.exceptions import DvcException
    from dvc.repo import Repo as _DvcRepo

    _HAS_DVC = True
except Exception:  # pragma: no cover - optional
    DvcException = Exception  # type: ignore
    _DvcRepo = None  # type: ignore
    _HAS_DVC = False

try:  # optional ray dependency
    import ray  # type: ignore

    _HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore
    _HAS_RAY = False


logger = logging.getLogger(__name__)

SEQUENCE_MODEL_TYPES = {"tabtransformer", "tcn"}


def _compute_time_decay_weights(
    event_times: np.ndarray | Sequence[object] | None, *, half_life_days: float
) -> np.ndarray:
    """Return exponential decay weights based on ``event_times``.

    Parameters
    ----------
    event_times:
        Sequence of event timestamps.  Values are coerced to ``datetime64``
        before computing ages relative to the most recent timestamp.
    half_life_days:
        Half-life in days controlling the decay rate.  When the value is
        non-positive or ``event_times`` is empty an array of ones is returned.
    """

    if event_times is None:
        return np.ones(0, dtype=float)

    dt_index = pd.to_datetime(pd.Index(event_times), errors="coerce", utc=True)
    try:
        dt_index = dt_index.tz_convert(None)
    except AttributeError:  # pragma: no cover - defensive for non-index inputs
        dt_index = pd.to_datetime(dt_index, errors="coerce")
    times = dt_index.to_numpy(dtype="datetime64[ns]")
    if times.size == 0:
        return np.ones(0, dtype=float)
    if half_life_days <= 0:
        return np.ones(times.size, dtype=float)

    mask = ~np.isnat(times)
    weights = np.ones(times.size, dtype=float)
    if not mask.any():
        return weights

    ref_time = times[mask].max()
    age_seconds = (ref_time - times[mask]).astype("timedelta64[s]").astype(float)
    age_days = age_seconds / (24 * 3600)
    decay = np.power(0.5, age_days / half_life_days)
    weights[mask] = decay
    if (~mask).any():
        fill_value = float(decay.min()) if decay.size else 1.0
        weights[~mask] = fill_value
    return weights


def _compute_volatility_scaler(
    profits: np.ndarray, *, window: int = 50
) -> np.ndarray:
    """Compute inverse volatility scaling factors for ``profits``."""

    if profits.size == 0:
        return np.ones(0, dtype=float)

    series = pd.Series(profits, dtype=float)
    vol = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
    scaler = 1.0 / (1.0 + vol.to_numpy(dtype=float))
    return np.clip(scaler, a_min=1e-6, a_max=None)


def _compute_profit_weights(profits: np.ndarray) -> np.ndarray:
    """Return base sample weights derived from trade profits."""

    if profits.size == 0:
        return np.ones(0, dtype=float)

    weights = np.abs(np.asarray(profits, dtype=float))
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    positive = weights > 0
    if positive.any():
        min_positive = float(weights[positive].min())
        weights = np.where(positive, weights, min_positive)
    else:
        weights = np.ones_like(weights)
    return weights


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Normalise ``weights`` to have a mean of one."""

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean()
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(arr, dtype=float)
    return arr / mean


def _summarise_weights(weights: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for ``weights`` suitable for logging."""

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _build_sample_weights(
    profits: np.ndarray,
    event_times: np.ndarray | Sequence[object] | None,
    *,
    half_life_days: float,
    use_volatility: bool,
) -> np.ndarray:
    """Combine profit, time-decay and volatility based weights."""

    base = _compute_profit_weights(profits)
    if event_times is not None and base.size:
        decay = _compute_time_decay_weights(event_times, half_life_days=half_life_days)
        if decay.size == base.size:
            base = base * decay
    if use_volatility and base.size:
        scaler = _compute_volatility_scaler(profits)
        if scaler.size == base.size:
            base = base * scaler
    return base


def _serialize_mlflow_param(value: object) -> str:
    """Convert arbitrary parameter values into a string for MLflow logging."""

    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return np.array2string(np.asarray(value), separator=",")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(_serialize_mlflow_param(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(
            {str(k): _serialize_mlflow_param(v) for k, v in value.items()},
            sort_keys=True,
        )
    return str(value)


def _version_artifacts_with_dvc(repo_root: Path | None, targets: Sequence[Path]) -> None:
    """Register the provided targets with a DVC repository if available."""

    if not (_HAS_DVC and repo_root):
        return
    repo_root = Path(repo_root).resolve()
    if not (repo_root / ".dvc").exists():
        logger.debug("DVC root %s is not initialized; skipping versioning", repo_root)
        return
    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        with _DvcRepo(str(repo_root)) as repo:  # type: ignore[misc]
            for target in targets:
                target_path = Path(target).resolve()
                if not target_path.exists():
                    continue
                try:
                    rel_target = target_path.relative_to(repo_root)
                except ValueError:
                    logger.debug(
                        "Skipping DVC registration for %s outside of repo %s",
                        target_path,
                        repo_root,
                    )
                    continue
                try:
                    repo.add([str(rel_target)])
                except DvcException:  # pragma: no cover - best effort logging
                    logger.debug(
                        "DVC reported that %s is already tracked; skipping", rel_target
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to add %s to DVC repository %s", rel_target, repo_root
                    )
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to open DVC repository at %s", repo_root)
    finally:
        os.chdir(cwd)


if _HAS_NUMBA:

    @njit
    def _max_drawdown_nb(returns: np.ndarray) -> float:
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            cum += r
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @njit
    def _var_95_nb(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        sorted_r = np.sort(returns)
        idx = int(0.05 * (sorted_r.size - 1))
        return -sorted_r[idx]

    def _max_drawdown(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        return float(_max_drawdown_nb(returns))

    def _var_95(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        return float(_var_95_nb(returns))

else:  # pragma: no cover - fallback without numba

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
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        packages = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if result.returncode != 0 and not packages:
            logger.exception(
                "pip freeze returned non-zero exit code %s", result.returncode
            )
    except Exception:  # pragma: no cover - pip not available
        logger.exception("Failed to execute pip freeze; falling back to metadata")
        packages = []
    if not packages:
        packages = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib_metadata.distributions()
        )
    dep_path = out_dir / "dependencies.txt"
    dep_path.write_text("\n".join(packages) + ("\n" if packages else ""))
    return dep_path


def _serialise_metric_values(obj: object) -> object:
    """Recursively convert numpy types within ``obj`` to JSON-friendly values."""

    if isinstance(obj, dict):
        return {k: _serialise_metric_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise_metric_values(v) for v in obj]
    if isinstance(obj, np.generic):  # includes scalar numpy types
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _trial_logger(
    csv_path: Path,
) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
    """Return a callback that appends trial information to ``csv_path``."""

    def _callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        row = {
            "trial": trial.number,
            **trial.params,
            "seed": trial.user_attrs.get("seed"),
            "profit": trial.values[0],
            "sharpe": trial.values[1],
            "max_drawdown": trial.values[2],
        }
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    return _callback


def _objective_factory(
    max_drawdown: float | None, var_limit: float | None
) -> Callable[[optuna.trial.Trial], tuple[float, float, float]]:
    """Return a multi-objective that includes risk penalties."""

    def _objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        seed = trial.suggest_int("seed", 0, 9999)
        trial.set_user_attr("seed", seed)
        x = trial.suggest_float("x", -10.0, 10.0)
        rng = np.random.default_rng(seed)
        noise = rng.normal()
        risk = abs(x) / 10.0
        trial.set_user_attr("max_drawdown", risk)
        trial.set_user_attr("var_95", risk)
        profit = -((x - 2) ** 2) + noise
        if max_drawdown is not None and risk > max_drawdown:
            profit -= risk - max_drawdown
        if var_limit is not None and risk > var_limit:
            profit -= risk - var_limit
        sharpe = profit / (risk + 1e-6)
        return profit, sharpe, risk

    return _objective


def run_optuna(
    n_trials: int = 10,
    csv_path: Path | str = "hyperparams.csv",
    model_json_path: Path | str = "model.json",
    *,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
) -> optuna.study.Study:
    """Run a small Optuna study and record trial information."""

    csv_path = Path(csv_path)
    model_json_path = Path(model_json_path)

    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"], sampler=sampler
    )
    objective = _objective_factory(max_drawdown, var_limit)
    study.optimize(objective, n_trials=n_trials, callbacks=[_trial_logger(csv_path)])

    def _select_trial() -> optuna.trial.FrozenTrial:
        candidates = [t for t in study.best_trials]
        if max_drawdown is not None:
            candidates = [t for t in candidates if t.values[2] <= max_drawdown]
        if var_limit is not None:
            candidates = [
                t
                for t in candidates
                if t.user_attrs.get("var_95", t.values[2]) <= var_limit
            ]
        if not candidates:
            candidates = list(study.best_trials)
        return max(candidates, key=lambda t: (t.values[0], t.values[1]))

    best = _select_trial()
    relative_csv = os.path.relpath(csv_path, model_json_path.parent)
    risk = best.user_attrs.get("max_drawdown", 0.0)
    model_data = {
        "metadata": {
            "hyperparam_log": relative_csv,
            "selected_trial": {
                "number": best.number,
                "profit": best.values[0],
                "sharpe": best.values[1],
                "max_drawdown": best.values[2],
            },
        },
        "risk_params": {"max_drawdown": max_drawdown, "var_limit": var_limit},
        "risk_metrics": {
            "max_drawdown": risk,
            "var_95": best.user_attrs.get("var_95", risk),
        },
    }
    model_json_path.write_text(json.dumps(model_data))

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
    complexity_penalty: float = 0.1,
    dvc_repo: Path | str | None = None,
    config_hash: str | None = None,
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
    chosen_action: tuple[tuple[str, ...], str] | None = None
    if controller is not None:
        controller.model_path = out_dir / "model.json"
        if not reuse_controller:
            controller.reset()
        chosen_action, _ = controller.sample_action()
        features = list(chosen_action[0])
        model_type = chosen_action[1]

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
    configure_cache(
        FeatureConfig(cache_dir=cache_dir, enabled_features=set(features or []))
    )
    memory = Memory(str(cache_dir) if cache_dir else None, verbose=0)
    _classification_metrics_cached = memory.cache(_classification_metrics)
    _hrp_cached = memory.cache(hierarchical_risk_parity)
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
        elif sequence_model:
            gpu_kwargs.update({"device": "cuda"})
    fs_repo = Path(__file__).resolve().parents[1] / "feature_store" / "feast_repo"
    span_ctx = tracer.start_as_current_span("feature_extraction")
    span_ctx.__enter__()
    y_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    profit_list: list[np.ndarray] = []
    event_time_list: list[np.ndarray] = []
    returns_frames: list[pd.DataFrame] = []
    event_times: np.ndarray = np.array([], dtype="datetime64[ns]")
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
                    chunk, feature_names, n_jobs=n_jobs, model_json=model_json
                )
            FeatureSchema.validate(chunk[feature_names], lazy=True)
            if label_col is None:
                label_col = next(
                    (c for c in chunk.columns if c.startswith("label")), None
                )
                if label_col is None:
                    raise ValueError("no label column found")
            X_list.append(chunk[feature_names].fillna(0.0).to_numpy(dtype=float))
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
                df, feature_names, n_jobs=n_jobs, model_json=model_json
            )
        FeatureSchema.validate(df[feature_names], lazy=True)
        if _HAS_DASK and isinstance(df, dd.DataFrame):
            df = df.compute()
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
        elif _HAS_POLARS and isinstance(df, pl.DataFrame):
            X = df.select(feature_names).fill_null(0.0).to_numpy().astype(float)
            if "profit" in df.columns:
                profits = df["profit"].to_numpy().astype(float)
                label_col = next((c for c in df.columns if str(c).startswith("label")), None)
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
                returns_df = (
                    df[ret_cols].to_pandas() if ret_cols else None
                )
            else:
                label_col = next((c for c in df.columns if c.startswith("label")), None)
                if label_col is None:
                    raise ValueError("no label column found")
                y = df[label_col].to_numpy().astype(float)
                profits = np.zeros_like(y)
                has_profit = False
                event_times = (
                    pd.to_datetime(df["event_time"].to_numpy(), errors="coerce").to_numpy()
                    if "event_time" in df.columns
                    else np.full(
                        df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
                    )
                )
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
    weight_times = event_times if event_times.size else None
    sample_weight = _build_sample_weights(
        profits,
        weight_times,
        half_life_days=half_life_days,
        use_volatility=vol_weight,
    )
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
    sequence_data: np.ndarray | None = None
    if sequence_model:
        if X.shape[0] < window_length:
            raise ValueError("Not enough samples for the requested window length")
        seq_list = [
            X[i - window_length + 1 : i + 1]
            for i in range(window_length - 1, X.shape[0])
        ]
        sequence_data = np.stack(seq_list, axis=0).astype(float)
        X = X[window_length - 1 :]
        y = y[window_length - 1 :]
        profits = profits[window_length - 1 :]
        sample_weight = sample_weight[window_length - 1 :]
        if R is not None:
            R = R[window_length - 1 :]
        if returns_df is not None:
            returns_df = returns_df.iloc[window_length - 1 :].reset_index(drop=True)
    else:
        sequence_data = None

    sample_weight = _normalise_weights(sample_weight)
    weight_stats = _summarise_weights(sample_weight)
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
        objective_metric_map = {
            "profit": "profit",
            "net_profit": "profit",
            "sharpe": "sharpe_ratio",
            "sortino": "sortino_ratio",
        }
        objective_key = objective_metric_map.get(threshold_objective, "profit")
        def _evaluate_thresholds_for_fold(
            fold_preds: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
        ) -> tuple[
            float, dict[str, object], list[tuple[int, dict[str, object]]]
        ]:
            if not fold_preds:
                return 0.5, {}, []
            fold_preds = sorted(fold_preds, key=lambda x: x[0])
            y_all = np.concatenate([fp[1] for fp in fold_preds])
            prob_all = np.concatenate([fp[2] for fp in fold_preds])
            profit_all = np.concatenate([fp[3] for fp in fold_preds])
            if not np.any(np.abs(profit_all) > 0):
                candidates = np.asarray([0.5], dtype=float)
            else:
                base_grid = (
                    threshold_grid_values
                    if threshold_grid_values is not None
                    else np.asarray([], dtype=float)
                )
                finite_probs = prob_all[np.isfinite(prob_all)]
                candidate = np.concatenate(
                    [
                        base_grid,
                        finite_probs,
                        np.asarray([0.0, 0.5, 1.0], dtype=float),
                    ]
                )
                candidate = candidate[(candidate >= 0.0) & (candidate <= 1.0)]
                if candidate.size == 0:
                    candidates = np.asarray([0.5], dtype=float)
                else:
                    candidates = np.unique(candidate)
                    if candidates.size > 512:
                        idx = np.linspace(0, candidates.size - 1, 512, dtype=int)
                        candidates = candidates[idx]
            best_metrics: dict[str, object] | None = None
            best_threshold = float(candidates[0])
            best_obj = -np.inf
            for thr in candidates:
                returns_all = profit_all * (prob_all >= thr)
                metrics_all = _classification_metrics_cached(
                    y_all, prob_all, returns_all, selected=metric_names
                )
                metrics_all["max_drawdown"] = _max_drawdown(returns_all)
                metrics_all["var_95"] = _var_95(returns_all)
                metrics_all["threshold"] = float(thr)
                metrics_all["threshold_objective"] = threshold_objective
                obj_val = metrics_all.get(objective_key)
                if isinstance(obj_val, (int, float)) and not np.isnan(obj_val):
                    obj_score = float(obj_val)
                else:
                    obj_score = -np.inf
                if obj_score > best_obj or (
                    np.isfinite(obj_score)
                    and np.isclose(obj_score, best_obj)
                    and float(thr) < best_threshold
                ):
                    best_obj = obj_score
                    best_threshold = float(thr)
                    best_metrics = metrics_all
            if best_metrics is None:
                best_metrics = {
                    "threshold": best_threshold,
                    "threshold_objective": threshold_objective,
                }
            fold_metrics: list[tuple[int, dict[str, object]]] = []
            for fold_idx, y_val, prob_val, prof_val in fold_preds:
                returns_fold = prof_val * (prob_val >= best_threshold)
                metrics_fold = _classification_metrics_cached(
                    y_val, prob_val, returns_fold, selected=metric_names
                )
                metrics_fold["max_drawdown"] = _max_drawdown(returns_fold)
                metrics_fold["var_95"] = _var_95(returns_fold)
                metrics_fold["threshold"] = float(best_threshold)
                metrics_fold["threshold_objective"] = threshold_objective
                fold_metrics.append((fold_idx, metrics_fold))
            return best_threshold, best_metrics, fold_metrics

        if distributed and not _HAS_RAY:
            raise RuntimeError("ray is required for distributed execution")
        for params in param_grid:
            fold_predictions: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            if distributed and _HAS_RAY:
                X_ref = ray.put(model_inputs)
                y_ref = ray.put(y)
                profits_ref = ray.put(profits)
                weight_ref = ray.put(sample_weight)
                R_ref = ray.put(R) if model_type == "moe" else None

                @ray.remote
                def _run_fold(tr_idx, val_idx, fold):
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
                        if model_type == "moe" or sequence_model:
                            builder_kwargs["grad_clip"] = grad_clip
                        if meta_init is not None:
                            builder_kwargs["init_weights"] = meta_init
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
                        if model_type == "moe" or sequence_model:
                            builder_kwargs["grad_clip"] = grad_clip
                        if meta_init is not None:
                            builder_kwargs["init_weights"] = meta_init
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
            threshold_value, combined_metrics, fold_metric_entries = _evaluate_thresholds_for_fold(
                fold_predictions
            )
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
            if model_type == "moe" or sequence_model:
                builder_kwargs["grad_clip"] = grad_clip
            if meta_init is not None:
                builder_kwargs["init_weights"] = meta_init
            model_data, predict_fn = builder(
                model_inputs,
                y,
                **builder_kwargs,
            )

        # --- SHAP based feature selection ---------------------------------
        shap_threshold = float(kwargs.get("shap_threshold", 0.0))
        if not sequence_model:
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
                            model_inputs = X
                            builder_kwargs = dict(
                                **gpu_kwargs, **(best_params or {}), **extra_model_params
                            )
                            builder_kwargs["sample_weight"] = sample_weight
                            if model_type == "moe" or sequence_model:
                                builder_kwargs["grad_clip"] = grad_clip
                            model_data, predict_fn = builder(
                                model_inputs,
                                y,
                                **builder_kwargs,
                            )
            except Exception:  # pragma: no cover - shap is optional
                logger.exception("Failed to compute SHAP values")
        model_obj = getattr(predict_fn, "model", None)

        # --- Probability calibration ---------------------------------------
        calibration_info: dict[str, object] | None = None
        if model_type != "moe" and not sequence_model:
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
        feature_metadata = [FeatureMetadata(original_column=fn) for fn in feature_names]
        model = {
            "feature_names": feature_names,
            "feature_metadata": feature_metadata,
            **model_data,
            "model_type": model_type,
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
        serialised_cv_metrics = _serialise_metric_values(metrics)
        session_cv_metrics = _serialise_metric_values(metrics)
        serialised_fold_metrics = [
            _serialise_metric_values(m) for m in best_fold_metrics
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
            model["session_models"] = {
                "asian": {k: model[k] for k in sm_keys if k in model}
            }
        model["cv_metrics"] = serialised_cv_metrics
        model["threshold"] = float(selected_threshold)
        model["decision_threshold"] = float(selected_threshold)
        model["threshold_objective"] = threshold_objective
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
        deps_file = _write_dependency_snapshot(out_dir)
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
        _version_artifacts_with_dvc(
            dvc_repo_path,
            [
                data_dir,
                out_dir / "model.json",
                out_dir / "data_hashes.json",
            ],
        )
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
            if regime_features is not None:
                base_params["regime_features"] = list(regime_features)
            if dvc_repo_path is not None:
                base_params["dvc_repo"] = dvc_repo_path
            mlflow.log_params({
                k: _serialize_mlflow_param(v) for k, v in base_params.items()
            })
            if best_params:
                mlflow.log_params(
                    {
                        f"hp_{k}": _serialize_mlflow_param(v)
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
            model_uri = mlflow.get_artifact_uri("model/model.json")
            data_hash_uri = mlflow.get_artifact_uri("model/data_hashes.json")
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("data_hashes_uri", data_hash_uri)
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
    if feature_names and len(feature_names) == features.shape[1]:
        df = pd.DataFrame(features, columns=feature_names)
        FeatureSchema.validate(df, lazy=True)
    clip_low = np.asarray(params.get("clip_low", model.get("clip_low", [])), dtype=float)
    clip_high = np.asarray(
        params.get("clip_high", model.get("clip_high", [])), dtype=float
    )
    if clip_low.size and clip_high.size and clip_low.shape == clip_high.shape:
        features = np.clip(features, clip_low, clip_high)

    mean = np.asarray(params.get("feature_mean", model.get("feature_mean", [])), dtype=float)
    std = np.asarray(params.get("feature_std", model.get("feature_std", [])), dtype=float)
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
    if _HAS_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
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
        meta_weights=train_cfg.meta_weights,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
