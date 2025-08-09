"""Arrow and Avro schemas for trade and metric records."""

from pathlib import Path
import json

from .trades import TRADE_SCHEMA
from .metrics import METRIC_SCHEMA

SCHEMA_DIR = Path(__file__).resolve().parent
with (SCHEMA_DIR / "trade.avsc").open() as f:
    TRADE_AVRO_SCHEMA = json.load(f)
with (SCHEMA_DIR / "metric.avsc").open() as f:
    METRIC_AVRO_SCHEMA = json.load(f)

__all__ = [
    "TRADE_SCHEMA",
    "METRIC_SCHEMA",
    "TRADE_AVRO_SCHEMA",
    "METRIC_AVRO_SCHEMA",
]
