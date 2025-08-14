"""Arrow and Pydantic schemas for periodic metric updates."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
import pyarrow as pa


class MetricEvent(BaseModel):
    """Pydantic model describing a metric update."""

    schema_version: int
    time: datetime
    magic: int
    win_rate: float
    avg_profit: float
    trade_count: int
    drawdown: float
    sharpe: float
    file_write_errors: int
    socket_errors: int
    book_refresh_seconds: int

    class Config:
        extra = "ignore"


METRIC_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
    ("trace_id", pa.string()),
    ("time", pa.string()),
    ("magic", pa.int32()),
    ("win_rate", pa.float64()),
    ("avg_profit", pa.float64()),
    ("trade_count", pa.int32()),
    ("drawdown", pa.float64()),
    ("sharpe", pa.float64()),
    ("file_write_errors", pa.int32()),
    ("socket_errors", pa.int32()),
    ("book_refresh_seconds", pa.int32()),
])
