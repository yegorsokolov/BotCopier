"""Feast feature view definitions for core BotCopier features."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

# Define the raw data source.  The path is resolved relative to this file so that
# the feature repository can be moved without breaking the configuration.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "trades.parquet"

trade_source = FileSource(
    path=str(DATA_PATH),
    timestamp_field="event_time",
    created_timestamp_column="created",
)

# Entity representing a traded symbol.
symbol = Entity(name="symbol", join_keys=["symbol"])

# Core set of engineered features used across training and inference.  These are
# lightweight temporal encodings that are always present in the raw logs.
trade_features_view = FeatureView(
    name="trade_features",
    entities=[symbol],
    ttl=timedelta(days=1),
    schema=[
        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),
        Field(name="dow_sin", dtype=Float32),
        Field(name="dow_cos", dtype=Float32),
        Field(name="month_sin", dtype=Float32),
        Field(name="month_cos", dtype=Float32),
        Field(name="dom_sin", dtype=Float32),
        Field(name="dom_cos", dtype=Float32),
    ],
    online=True,
    source=trade_source,
)

# Export list of feature names for consumers.
FEATURE_COLUMNS = [field.name for field in trade_features_view.schema]
