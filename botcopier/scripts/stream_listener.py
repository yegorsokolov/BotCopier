#!/usr/bin/env python3
"""Consume WAL entries, validate schema versions and ship logs to GitHub.

The real project streams trade and metric events from MetaTrader.  For the
tests we keep the implementation intentionally small but still provide the
pieces required by downstream tools:

* ``process_trade`` and ``process_metric`` validate incoming messages and append
  them to CSV logs.
* ``capture_system_metrics`` records basic host telemetry.
* ``validate_and_persist`` stores validated Arrow tables as a Parquet dataset.
* ``consume_wal`` loads JSON entries from a write‑ahead log, validates the
  schema version and persists them.
* ``upload_logs`` commits any new log files to a Git repository and pushes to
  GitHub when a ``GITHUB_TOKEN`` is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import format_span_id, format_trace_id

try:  # optional system metrics
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil not installed
    from types import SimpleNamespace

    class _PSUtilStub:
        def cpu_percent(self, interval=None):
            return 0.0

        def virtual_memory(self):
            return SimpleNamespace(percent=0.0)

        def net_io_counters(self):
            return SimpleNamespace(bytes_sent=0, bytes_recv=0)

    psutil = _PSUtilStub()  # type: ignore

try:  # optional heavy dependency
    import pyarrow as pa  # type: ignore
    import pyarrow.flight as flight  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - pyarrow not installed
    pa = None  # type: ignore
    pq = None  # type: ignore
    flight = None  # type: ignore

logger = logging.getLogger(__name__)

if pa is not None:  # pragma: no cover - optional dependency guards
    _PARQUET_EXCEPTIONS = (OSError, pa.ArrowInvalid, pa.ArrowIOError)
else:  # pragma: no cover - pyarrow missing
    _PARQUET_EXCEPTIONS = (OSError,)

if flight is not None:  # pragma: no cover - optional dependency guards
    _FLIGHT_EXCEPTIONS = (flight.FlightError, OSError)
else:  # pragma: no cover - pyarrow.flight missing
    _FLIGHT_EXCEPTIONS = (OSError,)

SCHEMA_VERSION = 1


def _snake(name: str) -> str:
    out = []
    for c in name:
        if c.isupper():
            out.append("_")
            out.append(c.lower())
        else:
            out.append(c)
    return "".join(out).lstrip("_")


TRADE_FIELDS: Dict[str, type] = {
    "eventId": int,
    "eventTime": str,
    "brokerTime": str,
    "localTime": str,
    "action": str,
    "ticket": int,
    "magic": int,
    "source": str,
    "symbol": str,
    "orderType": int,
    "lots": float,
    "price": float,
    "sl": float,
    "tp": float,
    "profit": float,
    "comment": str,
    "remainingLots": float,
}

METRIC_FIELDS: Dict[str, type] = {
    "time": str,
    "magic": int,
    "winRate": float,
    "avgProfit": float,
    "tradeCount": int,
    "drawdown": float,
    "sharpe": float,
    "fileWriteErrors": int,
    "socketErrors": int,
    "queueBacklog": int,
    "bookRefreshSeconds": int,
}

if pa is not None:  # pragma: no cover - exercised in integration tests
    METRIC_SCHEMA = pa.schema(
        [
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
            ("queue_backlog", pa.int32()),
            ("book_refresh_seconds", pa.int32()),
        ]
    )
else:  # pragma: no cover - pyarrow missing
    METRIC_SCHEMA = None  # type: ignore

current_trace_id = ""
current_span_id = ""

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)


from contextlib import contextmanager


@contextmanager
def _span(name: str):
    with tracer.start_as_current_span(name) as span:
        ctx = span.get_span_context()
        global current_trace_id, current_span_id
        current_trace_id = format_trace_id(ctx.trace_id)
        current_span_id = format_span_id(ctx.span_id)
        yield


def append_csv(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def execute_limit_order(
    order: Dict[str, Any], bid: float, ask: float
) -> Dict[str, Any]:
    """Simulate execution of a limit ``order`` against current bid/ask prices.

    Orders that merely touch the limit price are partially filled while the
    remainder stays resting on the book.  Filled volume accumulates in
    ``filledLots`` and unfilled size is tracked via ``remainingLots``.
    """

    remaining = float(order.get("remainingLots", order.get("lots", 0.0)))
    side = int(order.get("orderType", 0))  # 0=buy, 1=sell
    price = float(order.get("price", 0.0))
    filled = 0.0

    if side == 0:  # buy
        if ask < price:
            filled = remaining
        elif ask == price:
            filled = remaining / 2.0
    else:  # sell
        if bid > price:
            filled = remaining
        elif bid == price:
            filled = remaining / 2.0

    remaining -= filled
    order["remainingLots"] = remaining
    order["filledLots"] = order.get("filledLots", 0.0) + filled
    return order


def _validate(fields: Dict[str, type], msg: Any, kind: str) -> Dict[str, Any] | None:
    data: Dict[str, Any] = {}
    for name, typ in fields.items():
        if not hasattr(msg, name):
            logger.warning(
                {"error": f"invalid {kind} event", "details": f"missing {name}"}
            )
            return None
        val = getattr(msg, name)
        if not isinstance(val, typ):
            logger.warning(
                {"error": f"invalid {kind} event", "details": f"field {name}"}
            )
            return None
        data[_snake(name)] = val
    schema_version = getattr(msg, "schemaVersion", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        logger.warning(
            {
                "error": "schema version mismatch",
                "expected": SCHEMA_VERSION,
                "got": schema_version,
            }
        )
        return None
    return data


def process_trade(msg) -> None:
    with _span("process_trade"):
        record = _validate(TRADE_FIELDS, msg, "trade")
        if record is None:
            return
        decision_id = getattr(msg, "decisionId", None)
        if decision_id in (None, 0):
            match = re.search(r"decision_id=(\d+)", record.get("comment", ""))
            if match:
                decision_id = int(match.group(1))
        record["decision_id"] = decision_id
        record["trace_id"] = current_trace_id
        record["span_id"] = current_span_id
        append_csv(Path("logs/trades_raw.csv"), record)


def process_metric(msg) -> None:
    with _span("process_metric"):
        record = _validate(METRIC_FIELDS, msg, "metric")
        if record is None:
            return
        record["trace_id"] = current_trace_id
        record["span_id"] = current_span_id
        append_csv(Path("logs/metrics.csv"), record)


def capture_system_metrics() -> None:
    record = {
        "trace_id": current_trace_id,
        "span_id": current_span_id,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "mem_percent": psutil.virtual_memory().percent,
        "bytes_sent": psutil.net_io_counters().bytes_sent,
        "bytes_recv": psutil.net_io_counters().bytes_recv,
    }
    append_csv(Path("logs/system_metrics.csv"), record)


def validate_and_persist(table, schema, dest: Path) -> bool:
    """Validate ``table`` against ``schema`` and append to a dataset."""
    if pa is None or pq is None:
        logger.warning({"error": "pyarrow not available"})
        return False
    if not table.schema.equals(schema):
        logger.warning(
            {
                "error": "schema mismatch",
                "expected": schema.to_string(),
                "got": table.schema.to_string(),
            }
        )
        return False
    try:
        dest.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(table, root_path=str(dest))
        return True
    except _PARQUET_EXCEPTIONS as exc:  # pragma: no cover - disk full etc.
        logger.exception(
            {
                "error": "persist failure",
                "details": str(exc),
                "path": str(dest),
            }
        )
        return False


def consume_wal(wal: Path, schema, dest: Path) -> None:
    """Read WAL entries from ``wal`` and persist validated rows."""
    if pa is None:
        logger.warning({"error": "pyarrow not available"})
        return
    if not wal.exists():
        return
    remaining: list[str] = []
    with wal.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning({"error": "invalid json", "line": line})
                continue
            if record.get("schema_version") != SCHEMA_VERSION:
                logger.warning(
                    {
                        "error": "schema version mismatch",
                        "got": record.get("schema_version"),
                    }
                )
                remaining.append(line)
                continue
            table = pa.Table.from_pylist([record], schema=schema)
            if not validate_and_persist(table, schema, dest):
                remaining.append(line)
    tmp = wal.with_suffix(".tmp")
    with tmp.open("w") as f:
        for line in remaining:
            f.write(line + "\n")
    tmp.replace(wal)


def consume_flight(path: str, schema, dest: Path, host: str, port: int) -> None:
    """Retrieve a record batch stream from an Arrow Flight server."""
    if pa is None or flight is None:
        logger.warning({"error": "pyarrow.flight not available"})
        return
    client = flight.FlightClient(f"grpc://{host}:{port}")
    ticket = flight.Ticket(path.encode())
    try:
        table = client.do_get(ticket).read_all()
    except _FLIGHT_EXCEPTIONS as exc:  # pragma: no cover - network issues
        logger.exception(
            {
                "error": "flight error",
                "details": str(exc),
                "endpoint": f"{host}:{port}",
                "path": path,
            }
        )
        return
    validate_and_persist(table, schema, dest)


def upload_logs(repo_path: Path, message: str) -> None:
    """Commit and push log files to a Git repository."""
    try:
        subprocess.run(["git", "-C", str(repo_path), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "commit", "-m", message], check=True
        )
        token = os.getenv("GITHUB_TOKEN")
        if token:
            subprocess.run(["git", "-C", str(repo_path), "push"], check=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover - git failure
        logger.warning({"error": "git push failed", "details": str(e)})


def main() -> None:  # pragma: no cover - simple CLI wrapper
    parser = argparse.ArgumentParser(description="Consume a Flight stream")
    parser.add_argument("dest", type=Path, help="output dataset directory")
    parser.add_argument("--flight-host", default="127.0.0.1")
    parser.add_argument("--flight-port", type=int, default=8815)
    parser.add_argument("--repo", type=Path, help="git repository to push logs")
    args = parser.parse_args()
    consume_flight(
        "metrics", METRIC_SCHEMA, args.dest, args.flight_host, args.flight_port
    )
    if args.repo:
        upload_logs(args.repo, "update logs")


if __name__ == "__main__":  # pragma: no cover
    main()
