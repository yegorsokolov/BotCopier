import numpy as np

from scripts.model_fitting import fit_logistic_regression


def test_fit_logistic_regression():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = fit_logistic_regression(X, y)
    preds = clf.predict(X)
    assert (preds == y).mean() >= 0.75
