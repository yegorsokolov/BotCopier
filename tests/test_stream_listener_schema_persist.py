import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import stream_listener as sl


def _metric_table():
    return pa.table(
        [
            pa.array([1], type=pa.int32()),
            pa.array(["2024-01-01T00:00:00"]),
            pa.array([1], type=pa.int32()),
            pa.array([0.5], type=pa.float64()),
            pa.array([0.1], type=pa.float64()),
            pa.array([10], type=pa.int32()),
            pa.array([0.2], type=pa.float64()),
            pa.array([1.0], type=pa.float64()),
            pa.array([0], type=pa.int32()),
            pa.array([0], type=pa.int32()),
            pa.array([5], type=pa.int32()),
        ],
        schema=sl.METRIC_SCHEMA,
    )


def test_validate_and_persist_roundtrip(tmp_path):
    table = _metric_table()
    dest = tmp_path / "metrics"
    assert sl.validate_and_persist(table, sl.METRIC_SCHEMA, dest)
    files = list(dest.glob("*.parquet"))
    assert files
    loaded = pq.read_table(dest)
    assert loaded.schema.equals(sl.METRIC_SCHEMA)
    assert loaded.to_pylist()[0]["trade_count"] == 10


def test_validate_and_persist_schema_mismatch(tmp_path, caplog):
    table = pa.table({"x": pa.array([1])})
    dest = tmp_path / "bad"
    with caplog.at_level(logging.WARNING):
        assert not sl.validate_and_persist(table, sl.METRIC_SCHEMA, dest)
    assert not dest.exists()
    assert "schema mismatch" in caplog.text
