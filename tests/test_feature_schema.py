import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pandera")

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
