import pytest

np = pytest.importorskip("numpy")
from botcopier.scripts import serve_model as sm


def test_mahalanobis_ood_flag():
    """Samples far from training mean should be flagged as OOD."""
    sm._configure_model(
        {
            "entry_coefficients": [1.0, 1.0],
            "entry_intercept": 0.0,
            "feature_names": ["f1", "f2"],
            "threshold": 0.0,
            "ood": {
                "mean": [0.0, 0.0],
                "covariance": [[1.0, 0.0], [0.0, 1.0]],
                "threshold": 1.0,
            },
        }
    )
    sm.OOD_COUNTER._value.set(0)
    pred = sm._predict_one([10.0, 10.0])
    assert pred == 0.0
    assert sm.OOD_COUNTER._value.get() == 1
