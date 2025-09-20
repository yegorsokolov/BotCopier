"""Regression tests ensuring SHAP explainers integrate with model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from botcopier.models.registry import get_model
from botcopier.shap_utils import mean_absolute_shap

pytest.importorskip("shap")


def _make_dataset(rng: np.random.Generator, n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_samples, 3))
    logits = 2.5 * X[:, 0] + 0.2 * rng.normal(size=n_samples)
    y = (logits > 0).astype(int)
    return X, y


def test_linear_model_shap_prefers_strong_feature() -> None:
    rng = np.random.default_rng(0)
    X, y = _make_dataset(rng)
    _, predict_fn = get_model("logreg")(X, y)

    mean_abs = mean_absolute_shap(predict_fn, X, model_type="logreg")

    assert mean_abs.shape == (X.shape[1],)
    assert int(np.argmax(mean_abs)) == 0
    assert np.all(np.isfinite(mean_abs))


def test_tree_model_shap_prefers_strong_feature() -> None:
    rng = np.random.default_rng(1)
    X, y = _make_dataset(rng)
    _, predict_fn = get_model("gradient_boosting")(X, y)

    mean_abs = mean_absolute_shap(
        predict_fn, X, model_type="gradient_boosting"
    )

    assert mean_abs.shape == (X.shape[1],)
    assert int(np.argmax(mean_abs)) == 0
    assert np.all(np.isfinite(mean_abs))
