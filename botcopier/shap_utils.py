"""Utility helpers shared across SHAP computations."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:  # scikit-learn is a hard dependency for training utilities
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover - defensive guard when sklearn missing
    Pipeline = None  # type: ignore

TREE_MODEL_TYPES = {
    "xgboost",
    "lightgbm",
    "random_forest",
    "gradient_boosting",
    "catboost",
}


def _extract_estimator(obj: Any) -> Any:
    """Return the underlying estimator stored on ``obj`` if present."""

    estimator = getattr(obj, "model", None)
    if estimator is None:
        estimator = obj
    return estimator


def _split_pipeline(estimator: Any) -> tuple[Any, Any | None]:
    """Return ``(final_estimator, preprocessor)`` for pipeline objects."""

    if Pipeline is None:
        return estimator, None
    if isinstance(estimator, Pipeline):
        steps = estimator.steps
        if not steps:
            return estimator, None
        if len(steps) == 1:
            return steps[0][1], None
        try:
            preprocessor = estimator[:-1]
        except TypeError:  # pragma: no cover - defensive for custom pipelines
            preprocessor = None
        final_estimator = steps[-1][1]
        return final_estimator, preprocessor
    return estimator, None


def _transform_background(
    preprocessor: Any | None, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ``preprocessor`` to ``X`` returning transformed background/data."""

    if preprocessor is None:
        return np.asarray(X), np.asarray(X)
    try:
        transformed = preprocessor.transform(X)
    except Exception:
        return np.asarray(X), np.asarray(X)
    return np.asarray(transformed), np.asarray(transformed)


def prepare_shap_inputs(
    model: Any,
    X: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return the estimator, background data and evaluation data for SHAP."""

    estimator = _extract_estimator(model)
    estimator, preprocessor = _split_pipeline(estimator)
    background, eval_data = _transform_background(preprocessor, X)
    return estimator, background, eval_data


def _shap_values_to_array(values: Any) -> np.ndarray:
    """Normalise SHAP outputs to a 2D array."""

    if isinstance(values, list):
        if not values:
            return np.zeros((0, 0), dtype=float)
        # assume binary classification when two outputs are present
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
    else:
        arr = np.asarray(getattr(values, "values", values))
        if arr.ndim == 3:
            arr = arr[..., 1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _make_explainer(
    estimator: Any,
    background: np.ndarray,
    *,
    model_type: str | None = None,
):
    """Create an appropriate SHAP explainer for ``estimator``."""

    import shap  # type: ignore

    if estimator is None:
        raise ValueError("SHAP explainer requires a fitted estimator")

    if model_type == "logreg" or hasattr(estimator, "coef_"):
        try:
            return shap.LinearExplainer(estimator, background)
        except Exception:
            pass
    if (
        model_type in TREE_MODEL_TYPES
        or hasattr(estimator, "get_booster")
        or hasattr(estimator, "feature_importances_")
    ):
        try:
            return shap.TreeExplainer(estimator)
        except Exception:
            pass
    return shap.Explainer(estimator, background)


def mean_absolute_shap(
    model: Any,
    X: np.ndarray,
    *,
    model_type: str | None = None,
) -> np.ndarray:
    """Return mean absolute SHAP values for ``model`` evaluated on ``X``."""

    estimator, background, eval_data = prepare_shap_inputs(model, np.asarray(X))
    explainer = _make_explainer(estimator, background, model_type=model_type)
    try:
        values = explainer.shap_values(eval_data)
    except AttributeError:
        values = explainer(eval_data)
    shap_arr = _shap_values_to_array(values)
    if shap_arr.size == 0:
        return shap_arr
    return np.abs(shap_arr).mean(axis=0)
