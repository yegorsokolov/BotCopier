#!/usr/bin/env python3
"""Utilities for inspecting feature importance and partial dependence.

The helpers defined here are light-weight wrappers around scikit-learn's
:func:`permutation_importance` and :class:`PartialDependenceDisplay`.  They are
used after a model has been trained to compute diagnostic plots which are saved
under ``reports/feature_analysis``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

DEFAULT_OUTPUT = Path("reports/feature_analysis")


def save_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    out_dir: Path | None = None,
    *,
    n_repeats: int = 10,
    random_state: int | None = 42,
) -> np.ndarray:
    """Compute and save permutation feature importance.

    Parameters
    ----------
    model:
        Fitted scikit-learn compatible estimator.
    X, y:
        Feature matrix and target array used for computing the importance.
    feature_names:
        Names of the features corresponding to the columns of ``X``.
    out_dir:
        Directory where the plot and ranks will be saved.  Defaults to
        ``reports/feature_analysis``.
    n_repeats, random_state:
        Passed directly to :func:`permutation_importance`.

    Returns
    -------
    numpy.ndarray
        Array of feature names ordered by decreasing importance.
    """

    if out_dir is None:
        out_dir = DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]
    ranks = np.asarray(feature_names)[sorted_idx]

    # Save ranks for later inspection.
    np.save(out_dir / "importance_ranks.npy", ranks)

    # Plot permutation importance.
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances[sorted_idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(ranks, rotation=90)
    ax.set_ylabel("Permutation importance")
    ax.set_xlabel("Feature")
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png")
    plt.close(fig)

    return ranks


def save_partial_dependence(
    model,
    X: np.ndarray,
    features: Iterable[int],
    feature_names: Sequence[str],
    out_dir: Path | None = None,
) -> np.ndarray:
    """Compute and save partial dependence plots for ``features``.

    Parameters
    ----------
    model:
        Fitted estimator.
    X:
        Feature matrix used to compute the partial dependence.
    features:
        Iterable of feature indices to analyse.
    feature_names:
        Names of all features.
    out_dir:
        Directory where the plot and arrays will be saved.  Defaults to
        ``reports/feature_analysis``.

    Returns
    -------
    numpy.ndarray
        Array of partial dependence values with shape ``(len(features), m)``
        where ``m`` is the number of grid points evaluated.
    """

    if out_dir is None:
        out_dir = DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    display = PartialDependenceDisplay.from_estimator(
        model, X, list(features), feature_names=feature_names
    )
    display.figure_.tight_layout()
    display.figure_.savefig(out_dir / "partial_dependence.png")
    plt.close(display.figure_)

    # Extract and stack the average partial dependence values.
    pd_arrays = np.vstack([b["average"].ravel() for b in display.pd_results])
    np.save(out_dir / "partial_dependence.npy", pd_arrays)

    return pd_arrays


__all__ = ["save_permutation_importance", "save_partial_dependence"]


if __name__ == "__main__":
    # Minimal CLI to showcase the helpers; primarily useful for manual use.
    import argparse
    import joblib
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description="Inspect feature behaviour")
    parser.add_argument("model", type=Path, help="Path to a saved sklearn model")
    parser.add_argument("data", type=Path, help="CSV file containing features and target")
    parser.add_argument(
        "--target", required=True, help="Name of the target column in the data"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Output directory"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df.pop(args.target).values
    X = df.values
    feature_names = list(df.columns)
    model = joblib.load(args.model)

    save_permutation_importance(model, X, y, feature_names, args.out)
    save_partial_dependence(model, X, range(len(feature_names)), feature_names, args.out)

