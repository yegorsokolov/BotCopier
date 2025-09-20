"""Evaluation helpers and Optuna integration for training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # optional optuna support
    import optuna

    HAS_OPTUNA = True
except ImportError:  # pragma: no cover - optional
    optuna = None  # type: ignore
    HAS_OPTUNA = False

try:  # optional numba support
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - optional

    def njit(*a, **k):  # pragma: no cover - simple stub
        def _inner(f):
            return f

        return _inner

    HAS_NUMBA = False

try:  # optional ray dependency
    import ray  # type: ignore

    HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore
    HAS_RAY = False


if HAS_NUMBA:

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

    def max_drawdown(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        return float(_max_drawdown_nb(returns))

    def var_95(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        return float(_var_95_nb(returns))

else:  # pragma: no cover - fallback without numba

    def max_drawdown(returns: np.ndarray) -> float:
        """Return the maximum drawdown of ``returns``."""

        if returns.size == 0:
            return 0.0
        cum = np.cumsum(returns, dtype=float)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd))

    def var_95(returns: np.ndarray) -> float:
        """Return the 95% Value at Risk of ``returns``."""

        if returns.size == 0:
            return 0.0
        return float(-np.quantile(returns, 0.05))


def serialise_metric_values(obj: object) -> object:
    """Recursively convert numpy types within ``obj`` to JSON-friendly values."""

    if isinstance(obj, dict):
        return {k: serialise_metric_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialise_metric_values(v) for v in obj]
    if isinstance(obj, np.generic):  # includes scalar numpy types
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def trial_logger(
    csv_path: Path,
) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
    """Return a callback that appends trial information to ``csv_path``."""

    if not HAS_OPTUNA:  # pragma: no cover - defensive guard
        raise RuntimeError("optuna is required to use trial_logger")

    def _callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        values = trial.values or (float("nan"), float("nan"), float("nan"))
        row = {
            "trial": trial.number,
            **trial.params,
            "seed": trial.user_attrs.get("seed"),
            "profit": values[0],
            "sharpe": values[1],
            "max_drawdown": values[2],
            "var_95": trial.user_attrs.get("var_95"),
        }
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    return _callback


def resolve_data_path(data_cfg: Mapping[str, Any]) -> Path:
    """Return the primary data path from ``data_cfg``."""

    for attr in ("data_dir", "data", "csv", "log_dir"):
        value = data_cfg.get(attr)
        if value is not None:
            return Path(value)
    raise ValueError(
        "Data configuration must define 'data_dir', 'data', or 'csv' for optuna runs"
    )


def suggest_model_params(
    trial: "optuna.trial.Trial", model_type: str
) -> dict[str, float | int | bool]:
    """Sample model-specific hyperparameters for ``model_type``."""

    params: dict[str, float | int | bool] = {}
    if model_type == "logreg":
        params["C"] = trial.suggest_float("logreg_C", 1e-3, 10.0, log=True)
        params["max_iter"] = trial.suggest_int("logreg_max_iter", 100, 600)
    elif model_type == "confidence_weighted":
        params["r"] = trial.suggest_float("cw_r", 0.25, 8.0, log=True)
    elif model_type == "gradient_boosting":
        params["learning_rate"] = trial.suggest_float(
            "gb_learning_rate", 0.01, 0.3, log=True
        )
        params["n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 250)
        params["max_depth"] = trial.suggest_int("gb_max_depth", 2, 6)
    elif model_type == "ensemble_voting":
        params["C"] = trial.suggest_float("ensemble_C", 0.1, 10.0, log=True)
        params["max_iter"] = trial.suggest_int("ensemble_max_iter", 100, 1000)
        params["include_xgboost"] = trial.suggest_categorical(
            "ensemble_include_xgb", [False, True]
        )
    elif model_type == "tabtransformer":
        params["epochs"] = trial.suggest_int("tabtransformer_epochs", 5, 20)
        params["batch_size"] = trial.suggest_categorical(
            "tabtransformer_batch_size", [32, 64, 128]
        )
        params["lr"] = trial.suggest_float("tabtransformer_lr", 1e-4, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float(
            "tabtransformer_weight_decay", 1e-5, 1e-2, log=True
        )
        params["dropout"] = trial.suggest_float("tabtransformer_dropout", 0.0, 0.5)
    elif model_type == "tcn":
        params["epochs"] = trial.suggest_int("tcn_epochs", 5, 20)
        params["batch_size"] = trial.suggest_categorical("tcn_batch_size", [16, 32, 64])
        params["lr"] = trial.suggest_float("tcn_lr", 1e-4, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float("tcn_weight_decay", 1e-5, 1e-2, log=True)
        params["dropout"] = trial.suggest_float("tcn_dropout", 0.0, 0.5)
        params["channels"] = trial.suggest_int("tcn_channels", 8, 64)
        params["kernel_size"] = trial.suggest_int("tcn_kernel", 2, 6)
    return params


__all__ = [
    "HAS_OPTUNA",
    "HAS_NUMBA",
    "HAS_RAY",
    "max_drawdown",
    "optuna",
    "ray",
    "resolve_data_path",
    "serialise_metric_values",
    "suggest_model_params",
    "trial_logger",
    "var_95",
]
