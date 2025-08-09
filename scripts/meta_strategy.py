#!/usr/bin/env python3
"""Train a meta-classifier that maps regime features to best base model."""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_meta_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "best_model",
) -> Dict[str, object]:
    """Train multinomial logistic regression to select best model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing regime features and labels of best model.
    feature_cols : List[str]
        Columns to use as features.
    label_col : str, default "best_model"
        Column containing the index of the best-performing base model.

    Returns
    -------
    Dict[str, object]
        Dictionary with feature names, gating coefficients and intercepts.
    """
    X = df[feature_cols].values
    y = df[label_col].values
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    clf.fit(X, y)
    coeffs = clf.coef_
    intercepts = clf.intercept_
    # For binary problems, sklearn returns a single row; create symmetric rows
    if coeffs.shape[0] == 1 and len(clf.classes_) == 2:
        coeffs = np.vstack([np.zeros_like(coeffs[0]), coeffs[0]])
        intercepts = np.array([0.0, intercepts[0]])
    return {
        "feature_names": feature_cols,
        "gating_coefficients": coeffs.tolist(),
        "gating_intercepts": intercepts.tolist(),
    }


def select_model(params: Dict[str, object], features: Dict[str, float]) -> int:
    """Select the best model given regime features.

    Parameters
    ----------
    params : dict
        Output from :func:`train_meta_model` containing coefficients.
    features : Dict[str, float]
        Mapping from feature name to value.

    Returns
    -------
    int
        Index of chosen base model.
    """
    names = params.get("feature_names", [])
    coeffs = np.array(params.get("gating_coefficients", []))
    intercepts = np.array(params.get("gating_intercepts", []))
    if coeffs.size == 0:
        return 0
    x = np.array([features.get(n, 0.0) for n in names])
    scores = intercepts + coeffs.dot(x)
    return int(np.argmax(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta gating model")
    parser.add_argument("data", help="CSV file with regime features and best_model column")
    parser.add_argument("out", help="Output JSON file for gating parameters")
    parser.add_argument("--features", nargs="+", required=True, help="Feature column names")
    parser.add_argument("--label", default="best_model", help="Label column name")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    params = train_meta_model(df, args.features, label_col=args.label)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    main()
