import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight=None,
    C: float = 1.0,
    class_weight=None,
    existing_model: dict | None = None,
) -> LogisticRegression:
    """Fit a logistic regression model and return the classifier."""
    clf = LogisticRegression(
        max_iter=200,
        C=C,
        warm_start=existing_model is not None,
        class_weight=class_weight,
    )
    if existing_model is not None:
        clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
        clf.coef_ = np.array([existing_model.get("coefficients", [])])
        clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


