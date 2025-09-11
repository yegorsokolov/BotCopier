import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluation import evaluate_model


def test_evaluate_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(max_iter=100).fit(X, y)
    metrics = evaluate_model(clf, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert "reliability_curve" in metrics
