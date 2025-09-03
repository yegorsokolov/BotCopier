"""Arrow and Pydantic schemas for decision events."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
import pyarrow as pa


class DecisionEvent(BaseModel):
    """Pydantic model describing a trading decision."""

    schema_version: int
    event_id: int
    timestamp: datetime
    model_version: str
    action: str
    probability: float
    sl_dist: float
    tp_dist: float
    model_idx: int
    regime: int
    chosen: int
    risk_weight: float
    variance: float
    lots_predicted: float
    executed_model_idx: int
    features: str
    trace_id: Optional[str] = ""
    span_id: Optional[str] = ""

    class Config:
        extra = "ignore"


DECISION_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
    ("event_id", pa.int32()),
    ("timestamp", pa.string()),
    ("model_version", pa.string()),
    ("action", pa.string()),
    ("probability", pa.float64()),
    ("sl_dist", pa.float64()),
    ("tp_dist", pa.float64()),
    ("model_idx", pa.int32()),
    ("regime", pa.int32()),
    ("chosen", pa.int32()),
    ("risk_weight", pa.float64()),
    ("variance", pa.float64()),
    ("lots_predicted", pa.float64()),
    ("executed_model_idx", pa.int32()),
    ("features", pa.string()),
    ("trace_id", pa.string()),
    ("span_id", pa.string()),
])

__all__ = ["DecisionEvent", "DECISION_SCHEMA"]
