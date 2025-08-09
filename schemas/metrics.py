"""Arrow schema for periodic metric updates."""

import pyarrow as pa

METRIC_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
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
