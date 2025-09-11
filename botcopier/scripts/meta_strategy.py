#!/usr/bin/env python3
"""Train a meta-classifier that maps regime features to best base model."""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier


def train_meta_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "best_model",
    params: Optional[Dict[str, object]] = None,
    use_partial_fit: bool = False,
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
    params : dict, optional
        Previously trained parameters to warm-start incremental training.
    use_partial_fit : bool, default False
        When ``True``, use :class:`~sklearn.linear_model.SGDClassifier` with
        ``partial_fit`` to update parameters incrementally.

    Returns
    -------
    Dict[str, object]
        Dictionary with feature names, gating coefficients and intercepts.
    """
    X = df[feature_cols].values
    y = df[label_col].values

    if use_partial_fit:
        clf = SGDClassifier(loss="log_loss", random_state=0)
        if params:
            try:
                coef = np.array(params.get("gating_coefficients", []))
                intercept = np.array(params.get("gating_intercepts", []))
                classes = np.array(params.get("classes", []))
                if coef.shape[0] == 2 and classes.size == 2:
                    coef = coef[1:2]
                    intercept = intercept[1:2]
                clf.coef_ = coef
                clf.intercept_ = intercept
                if classes.size:
                    clf.classes_ = classes
            except Exception:
                pass
        if not hasattr(clf, "classes_") or len(getattr(clf, "classes_", [])) == 0:
            classes = np.unique(y)
        else:
            classes = clf.classes_
        clf.partial_fit(X, y, classes=classes)
    else:
        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=200
        )
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
        "classes": clf.classes_.tolist(),
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


class RollingMetrics:
    """Maintain rolling profit and accuracy for base models.

    The metrics are updated using exponential moving averages so that recent
    performance has greater influence.  Weights for combining model outputs are
    computed from these metrics using an exponential transformation and
    normalisation.
    """

    def __init__(self, n_models: int, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.profit = np.zeros(n_models, dtype=float)
        self.accuracy = np.zeros(n_models, dtype=float)

    def update(self, model_idx: int, profit: float, correct: bool) -> None:
        """Update rolling statistics for ``model_idx``."""
        self.profit[model_idx] = (1 - self.alpha) * self.profit[
            model_idx
        ] + self.alpha * float(profit)
        self.accuracy[model_idx] = (1 - self.alpha) * self.accuracy[
            model_idx
        ] + self.alpha * (1.0 if correct else 0.0)

    def weights(self) -> np.ndarray:
        """Return exponential weights derived from recent performance."""
        score = self.profit + self.accuracy
        score -= score.max()
        exp_score = np.exp(score)
        if exp_score.sum() == 0:
            return np.ones_like(exp_score) / len(exp_score)
        return exp_score / exp_score.sum()

    def combine(self, outputs: Iterable[float]) -> float:
        """Combine ``outputs`` from base models using current weights."""
        preds = np.array(list(outputs), dtype=float)
        w = self.weights()[: len(preds)]
        return float(np.dot(w, preds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta gating model")
    parser.add_argument(
        "--data", help="CSV file with regime features and best_model column"
    )
    parser.add_argument("--out", help="Output JSON file for gating parameters")
    parser.add_argument("--features", nargs="+", help="Feature column names")
    parser.add_argument("--label", help="Label column name")
    args = parser.parse_args()

    from botcopier.config.settings import DataConfig, TrainingConfig, save_params

    data_cfg = DataConfig(
        **{k: getattr(args, k) for k in ["data", "out"] if getattr(args, k) is not None}
    )
    train_cfg = TrainingConfig(
        **{
            k: getattr(args, k)
            for k in ["features", "label"]
            if getattr(args, k) is not None
        }
    )
    save_params(data_cfg, train_cfg)

    df = pd.read_csv(data_cfg.data)
    params = train_meta_model(df, train_cfg.features, label_col=train_cfg.label)
    out_path = Path(data_cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    main()
