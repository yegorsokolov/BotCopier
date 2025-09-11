"""Central schema definitions for BotCopier datasets."""
from __future__ import annotations

import pyarrow as pa

# Schema for decision replay logs
DECISION_LOG_SCHEMA = pa.schema(
    [
        pa.field("decision_id", pa.int64()),
        pa.field("probability", pa.float64()),
        pa.field("f1", pa.float64()),
        pa.field("profit", pa.float64()),
    ]
)

# Trade log schema reused from existing definitions
try:  # pragma: no cover - avoid import issues during docs
    from schemas.trades import TRADE_SCHEMA as TRADE_LOG_SCHEMA
except Exception:  # pragma: no cover
    TRADE_LOG_SCHEMA = pa.schema([])

SCHEMAS = {
    "decisions": DECISION_LOG_SCHEMA,
    "trades": TRADE_LOG_SCHEMA,
}

__all__ = ["DECISION_LOG_SCHEMA", "TRADE_LOG_SCHEMA", "SCHEMAS"]
