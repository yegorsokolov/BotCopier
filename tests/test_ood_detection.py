import numpy as np
from botcopier.scripts import serve_model as sm


def test_mahalanobis_ood_flag():
    """Samples far from training mean should be flagged as OOD."""
    sm.MODEL = {"entry_coefficients": [1.0, 1.0], "entry_intercept": 0.0}
    sm.FEATURE_NAMES = ["f1", "f2"]
    sm.FEATURE_METADATA = []
    sm.OOD_MEAN = np.zeros(2)
    sm.OOD_INV = np.eye(2)
    sm.OOD_THRESHOLD = 1.0
    sm.OOD_COUNTER._value.set(0)
    pred = sm._predict_one([10.0, 10.0])
    assert pred == 0.0
    assert sm.OOD_COUNTER._value.get() == 1
