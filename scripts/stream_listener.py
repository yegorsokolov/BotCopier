#!/usr/bin/env python3
"""Listen for observer events from NATS JetStream and append them to CSV logs."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import pkgutil
from pathlib import Path
import sys

import asyncio
import nats
from pydantic import BaseModel, ValidationError
try:  # optional websocket client
    from websocket import create_connection, WebSocket  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    create_connection = None
    WebSocket = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    format_span_id,
    format_trace_id,
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
    set_span_in_context,
)

from proto import trade_event_pb2, metric_event_pb2

SCHEMA_VERSION = 1
TRADE_MSG = 0
METRIC_MSG = 1

LOG_FILES = {
    TRADE_MSG: Path("logs/trades_raw.csv"),
    METRIC_MSG: Path("logs/metrics.csv"),
}

RUN_INFO_PATH = Path("logs/run_info.json")
run_info_written = False

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "stream_listener")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
elif os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST") and JaegerExporter:
    provider.add_span_processor(
        BatchSpanProcessor(
            JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )
        )
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint)))
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {"level": record.levelname}
        if isinstance(record.msg, dict):
            log.update(record.msg)
        else:
            log["message"] = record.getMessage()
        if hasattr(record, "trace_id"):
            log["trace_id"] = format_trace_id(record.trace_id)
        if hasattr(record, "span_id"):
            log["span_id"] = format_span_id(record.span_id)
        return json.dumps(log)


logger = logging.getLogger(__name__)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# optional dashboard websocket connections
ws_trades: WebSocket | None = None
ws_metrics: WebSocket | None = None


class TradeEvent(BaseModel):
    schema_version: int
    event_id: int
    event_time: str
    broker_time: str
    local_time: str
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
    comment: str
    remaining_lots: float
    decision_id: int | None = None


class MetricEvent(BaseModel):
    schema_version: int
    time: str
    magic: int
    win_rate: float
    avg_profit: float
    trade_count: int
    drawdown: float
    sharpe: float
    file_write_errors: int
    socket_errors: int
    book_refresh_seconds: int


def append_csv(path: Path, record: dict) -> None:
    """Append ``record`` to ``path`` writing a header row when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def write_run_info() -> None:
    global run_info_written
    if run_info_written:
        return
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "libraries": sorted(m.name for m in pkgutil.iter_modules()),
    }
    RUN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_INFO_PATH.open("w") as f:
        json.dump(info, f, indent=2)
    run_info_written = True


def _get(msg, camel: str, snake: str | None = None):
    """Retrieve attribute ``camel``/``snake`` from ``msg`` supporting dicts."""
    if snake is None:
        snake = camel
    if isinstance(msg, dict):
        return msg.get(snake, msg.get(camel))
    return getattr(msg, camel, getattr(msg, snake, None))


def _context_from_ids(trace_id: str, span_id: str):
    """Build a span context from ``trace_id`` and ``span_id`` if provided."""
    if trace_id and span_id:
        try:
            ctx = SpanContext(
                trace_id=int(trace_id, 16),
                span_id=int(span_id, 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )
            return set_span_in_context(NonRecordingSpan(ctx))
        except ValueError:
            pass
    return None


def process_trade(msg) -> None:
    trace_id = _get(msg, "traceId", "trace_id") or ""
    span_id = ""
    try:
        comment = _get(msg, "comment") or ""
        if isinstance(comment, str) and comment.startswith("span="):
            parts = comment.split(";", 1)
            span_id = parts[0][5:]
            comment = parts[1] if len(parts) > 1 else ""
        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": _get(msg, "eventId", "event_id"),
            "event_time": _get(msg, "eventTime", "event_time"),
            "broker_time": _get(msg, "brokerTime", "broker_time"),
            "local_time": _get(msg, "localTime", "local_time"),
            "action": _get(msg, "action"),
            "ticket": _get(msg, "ticket"),
            "magic": _get(msg, "magic"),
            "source": _get(msg, "source"),
            "symbol": _get(msg, "symbol"),
            "order_type": _get(msg, "orderType", "order_type"),
            "lots": _get(msg, "lots"),
            "price": _get(msg, "price"),
            "sl": _get(msg, "sl"),
            "tp": _get(msg, "tp"),
            "profit": _get(msg, "profit"),
            "comment": comment,
            "remaining_lots": _get(msg, "remainingLots", "remaining_lots"),
            "decision_id": _get(msg, "decisionId", "decision_id"),
        }
        record = TradeEvent(**record).dict()
    except (AttributeError, ValidationError, TypeError) as e:
        logger.warning({"error": "invalid trade event", "details": str(e)})
        return
    ctx_in = _context_from_ids(trace_id, span_id)
    with tracer.start_as_current_span("process_event", context=ctx_in) as span:
        ctx = span.get_span_context()
        record.setdefault("trace_id", trace_id or format_trace_id(ctx.trace_id))
        record.setdefault("span_id", span_id or format_span_id(ctx.span_id))
        extra = {}
        try:
            extra["trace_id"] = int(record["trace_id"], 16)
            extra["span_id"] = int(record["span_id"], 16)
        except (KeyError, ValueError):
            pass
        logger.info(record, extra=extra)
        append_csv(LOG_FILES[TRADE_MSG], record)
        if ws_trades:
            try:
                ws_trades.send(json.dumps(record))
            except Exception:
                pass


def process_metric(msg) -> None:
    trace_id = _get(msg, "traceId", "trace_id") or ""
    span_id = _get(msg, "spanId", "span_id") or ""
    try:
        record = {
            "schema_version": SCHEMA_VERSION,
            "time": _get(msg, "time"),
            "magic": _get(msg, "magic"),
            "win_rate": _get(msg, "winRate", "win_rate"),
            "avg_profit": _get(msg, "avgProfit", "avg_profit"),
            "trade_count": _get(msg, "tradeCount", "trade_count"),
            "drawdown": _get(msg, "drawdown"),
            "sharpe": _get(msg, "sharpe"),
            "file_write_errors": _get(msg, "fileWriteErrors", "file_write_errors"),
            "socket_errors": _get(msg, "socketErrors", "socket_errors"),
            "book_refresh_seconds": _get(msg, "bookRefreshSeconds", "book_refresh_seconds"),
        }
        record = MetricEvent(**record).dict()
    except (AttributeError, ValidationError, TypeError) as e:
        logger.warning({"error": "invalid metric event", "details": str(e)})
        return
    ctx_in = _context_from_ids(trace_id, span_id)
    with tracer.start_as_current_span("process_metric", context=ctx_in) as span:
        ctx = span.get_span_context()
        record.setdefault("trace_id", trace_id or format_trace_id(ctx.trace_id))
        record["span_id"] = span_id or format_span_id(ctx.span_id)
        extra = {}
        try:
            extra["trace_id"] = int(record["trace_id"], 16)
            extra["span_id"] = int(record["span_id"], 16)
        except (KeyError, ValueError):
            pass
        logger.info(record, extra=extra)
        append_csv(LOG_FILES[METRIC_MSG], record)
        if ws_metrics:
            try:
                ws_metrics.send(json.dumps(record))
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description="Listen for observer events")
    p.add_argument("--servers", default="nats://127.0.0.1:4222", help="NATS server URLs")
    p.add_argument("--ws-url", help="Dashboard websocket base URL, e.g. ws://localhost:8000", default="")
    p.add_argument("--api-token", default=os.getenv("DASHBOARD_API_TOKEN", ""), help="API token for dashboard authentication")
    args = p.parse_args()

    if args.ws_url and create_connection:
        global ws_trades, ws_metrics
        try:
            headers = [f"Authorization: Bearer {args.api_token}"] if args.api_token else []
            ws_trades = create_connection(f"{args.ws_url}/ws/trades", header=headers)
            ws_metrics = create_connection(f"{args.ws_url}/ws/metrics", header=headers)
        except Exception:
            ws_trades = None
            ws_metrics = None

    async def _run() -> None:
        nc = await nats.connect(args.servers)
        js = nc.jetstream()

        async def trade_handler(msg):
            version = msg.data[0]
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %d mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                await msg.ack()
                return
            try:
                trade = trade_event_pb2.TradeEvent.FromString(msg.data[1:])
            except Exception:
                await msg.ack()
                return
            process_trade(trade)
            await msg.ack()

        async def metric_handler(msg):
            version = msg.data[0]
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %d mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                await msg.ack()
                return
            try:
                metric = metric_event_pb2.MetricEvent.FromString(msg.data[1:])
            except Exception:
                await msg.ack()
                return
            process_metric(metric)
            await msg.ack()

        await js.subscribe("trades", durable="stream_listener_trades", cb=trade_handler)
        await js.subscribe("metrics", durable="stream_listener_metrics", cb=metric_handler)

        write_run_info()
        for pth in LOG_FILES.values():
            pth.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.Future()

    try:
        asyncio.run(_run())
    finally:
        if ws_trades:
            try:
                ws_trades.close()
            except Exception:
                pass
        if ws_metrics:
            try:
                ws_metrics.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
