#!/usr/bin/env python3
"""Generate model explanation report combining SHAP, Integrated Gradients and
permutation importance.

The module exposes :func:`generate_explanations` which is used by the
training pipeline to create a lightweight report of feature importances.
The report is written as Markdown (and a companion HTML file) and
contains the top features ranked by the different attribution methods.

The implementation intentionally keeps dependencies minimal.  ``shap`` is
attempted but optional; for linear models a simple fallback based on the
model coefficients is employed.  Integrated Gradients are approximated for
linear models and return zero for unsupported estimators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from botcopier.shap_utils import mean_absolute_shap

# ---------------------------------------------------------------------------
# Helper utilities


def _linear_coefficients(model) -> np.ndarray | None:
    """Extract linear model coefficients if available."""
    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if coef is not None:
            return np.asarray(coef).ravel()
    if hasattr(model, "named_steps"):
        steps = getattr(model, "named_steps")
        if "logreg" in steps:
            return np.asarray(steps["logreg"].coef_).ravel()
    if hasattr(model, "base_estimator"):
        return _linear_coefficients(model.base_estimator)
    return None


def _shap_importance(model, X: np.ndarray) -> np.ndarray:
    """Compute mean absolute SHAP values for ``X``."""
    try:
        return mean_absolute_shap(model, X)
    except Exception:
        coef = _linear_coefficients(model)
        if coef is None:
            return np.zeros(X.shape[1], dtype=float)
        return np.abs(coef) * np.std(X, axis=0)


def _integrated_gradients(
    model, X: np.ndarray, baseline: np.ndarray | None
) -> np.ndarray:
    """Approximate Integrated Gradients for linear models."""
    coef = _linear_coefficients(model)
    if coef is None:
        return np.zeros(X.shape[1], dtype=float)
    if baseline is None:
        baseline = np.zeros_like(X)
    ig = (X - baseline) * coef
    return np.abs(ig).mean(axis=0)


def _permutation_importance(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=0, scoring="accuracy"
        )
        return np.abs(result.importances_mean)
    except Exception:
        return np.zeros(X.shape[1], dtype=float)


# ---------------------------------------------------------------------------
# Public API


def generate_explanations(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    out_file: Path,
) -> Path:
    """Generate an explanation report for ``model`` on ``X``.

    Parameters
    ----------
    model:
        Fitted estimator exposing ``predict`` and ``score``.
    X:
        Feature matrix used for training.
    y:
        Target vector.
    feature_names:
        Names corresponding to columns of ``X``.
    out_file:
        Destination path for the Markdown report.  An HTML companion with the
        same stem is also written.
    """

    shap_vals = _shap_importance(model, X)
    ig_vals = _integrated_gradients(model, X, baseline=X.mean(axis=0, keepdims=True))
    perm_vals = _permutation_importance(model, X, y)

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "shap": shap_vals,
            "integrated_gradients": ig_vals,
            "permutation_importance": perm_vals,
        }
    )
    df.sort_values("shap", ascending=False, inplace=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    md_table = df.to_markdown(index=False)
    out_file.write_text(f"# Model Explanation\n\n{md_table}\n")
    html_path = out_file.with_suffix(".html")
    html_path.write_text(f"<h1>Model Explanation</h1>\n{df.to_html(index=False)}")
    return out_file


# ---------------------------------------------------------------------------
# CLI entry point (best-effort)


def main() -> None:  # pragma: no cover - convenience wrapper
    p = argparse.ArgumentParser(description="Generate model explanation report")
    p.add_argument("model", type=Path, help="Path to model.json")
    p.add_argument(
        "data",
        type=Path,
        help="CSV file with training data containing label column and features",
    )
    p.add_argument("out", type=Path, help="Output report path (.md or .html)")
    args = p.parse_args()

    data = json.loads(args.model.read_text())
    feature_names = data.get("feature_names", [])
    if not feature_names:
        raise SystemExit("model.json missing feature_names")

    df = pd.read_csv(args.data)
    label_col = next((c for c in df.columns if c.startswith("label")), None)
    if label_col is None:
        raise SystemExit("no label column found in data")
    X = df[feature_names].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=float)

    # very small logistic regression reconstruction for CLI usage
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    coefficients = np.asarray(data.get("coefficients", []))
    intercept = float(data.get("intercept", 0.0))
    model = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression())])
    model.named_steps["logreg"].coef_ = coefficients.reshape(1, -1)
    model.named_steps["logreg"].intercept_ = np.array([intercept])
    model.named_steps["logreg"].classes_ = np.array([0, 1])
    model.named_steps["scale"].fit(np.zeros_like(X))

    generate_explanations(model, X, y, feature_names, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
