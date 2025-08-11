"""Arrow and Pydantic schemas for trade event records."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
import pyarrow as pa


class TradeEvent(BaseModel):
    """Pydantic model describing a trade event."""

    schema_version: int
    event_id: int
    event_time: datetime
    broker_time: Optional[datetime] = None
    local_time: Optional[datetime] = None
    action: str
    ticket: int
    magic: int
    source: str
    symbol: str
    order_type: int
    lots: float
    price: float
    sl: float
    tp: float
    profit: float
    comment: str = ""
    remaining_lots: float
    decision_id: Optional[int] = None

    class Config:
        extra = "ignore"


TRADE_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
    ("event_id", pa.int32()),
    ("trace_id", pa.string()),
    ("event_time", pa.string()),
    ("broker_time", pa.string()),
    ("local_time", pa.string()),
    ("action", pa.string()),
    ("ticket", pa.int32()),
    ("magic", pa.int32()),
    ("source", pa.string()),
    ("symbol", pa.string()),
    ("order_type", pa.int32()),
    ("lots", pa.float64()),
    ("price", pa.float64()),
    ("sl", pa.float64()),
    ("tp", pa.float64()),
    ("profit", pa.float64()),
    ("profit_after_trade", pa.float64()),
    ("spread", pa.float64()),
    ("comment", pa.string()),
    ("remaining_lots", pa.float64()),
    ("slippage", pa.float64()),
    ("volume", pa.int32()),
    ("open_time", pa.string()),
    ("book_bid_vol", pa.float64()),
    ("book_ask_vol", pa.float64()),
    ("book_imbalance", pa.float64()),
    ("sl_hit_dist", pa.float64()),
    ("tp_hit_dist", pa.float64()),
    ("decision_id", pa.int32()),
])
