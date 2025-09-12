import pandas as pd
import pandera as pa
import pytest

from botcopier.data.feature_schema import FeatureSchema


def test_type_violation():
    df = pd.DataFrame({"atr": ["bad"], "sl_dist_atr": [1.0]})
    with pytest.raises(pa.errors.SchemaErrors):
        FeatureSchema.validate(df, lazy=True)


def test_range_violation():
    df = pd.DataFrame({"atr": [1.0], "sl_dist_atr": [-1.0]})
    with pytest.raises(pa.errors.SchemaErrors):
        FeatureSchema.validate(df, lazy=True)
