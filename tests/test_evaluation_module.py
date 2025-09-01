import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluation import evaluate_model


def test_evaluate_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(max_iter=100).fit(X, y)
    acc = evaluate_model(clf, X, y)
    assert acc == 1.0
