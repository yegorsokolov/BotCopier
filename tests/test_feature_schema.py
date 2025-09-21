import sys
import types

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pandera")

training_mod = sys.modules.get("botcopier.training")
if training_mod is not None and not hasattr(training_mod, "__path__"):
    sys.modules.pop("botcopier.training", None)
    sys.modules.pop("botcopier.training.preprocessing", None)

if "gplearn" not in sys.modules:
    gplearn_mod = types.ModuleType("gplearn")
    gplearn_mod.genetic = types.SimpleNamespace(SymbolicTransformer=object)
    sys.modules["gplearn"] = gplearn_mod
    sys.modules["gplearn.genetic"] = gplearn_mod.genetic

from botcopier.data.feature_schema import FeatureSchema
from botcopier.training.pipeline import predict_expected_value


def test_type_violation():
    df = pd.DataFrame({"atr": ["bad"], "sl_dist_atr": [1.0]})
    with pytest.raises(pa.errors.SchemaErrors):
        FeatureSchema.validate(df, lazy=True)


def test_range_violation():
    df = pd.DataFrame({"atr": [1.0], "sl_dist_atr": [-1.0]})
    with pytest.raises(pa.errors.SchemaErrors):
        FeatureSchema.validate(df, lazy=True)


def test_predict_expected_value_schema_violation():
    model = {
        "feature_names": ["atr"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "coefficients": [1.0],
        "intercept": 0.0,
    }
    X = np.array([[-1.0]])  # atr must be >= 0
    with pytest.raises(pa.errors.SchemaErrors):
        predict_expected_value(model, X)


def test_predict_expected_value_feature_mismatch():
    model = {
        "feature_names": ["atr", "sl_dist_atr"],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "coefficients": [1.0, 1.0],
        "intercept": 0.0,
    }
    # Only provide a single column so the schema should reject the mismatch.
    X = np.array([[1.0]])
    with pytest.raises(ValueError):
        predict_expected_value(model, X)


def test_predict_expected_value_accepts_alias_metadata():
    model = {
        "feature_names": ["feat_a", "feat_b"],
        "feature_metadata": [
            {"original_column": "base"},
            {"original_column": "base"},
        ],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "coefficients": [0.6, -0.25],
        "intercept": 0.1,
    }
    X = np.array([[1.2], [-0.3], [0.0]], dtype=float)

    preds = predict_expected_value(model, X)

    features = np.column_stack([X[:, 0], X[:, 0]])
    logits = features @ np.array(model["coefficients"], dtype=float) + model["intercept"]
    expected = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(preds, expected)
