#!/usr/bin/env python3
"""Listen for observer events over ZeroMQ and append them to CSV logs."""
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

import capnp
import zmq
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
from opentelemetry.trace import format_span_id, format_trace_id

trade_capnp = capnp.load(str(Path(__file__).resolve().parents[1] / "proto" / "trade.capnp"))
metrics_capnp = capnp.load(str(Path(__file__).resolve().parents[1] / "proto" / "metrics.capnp"))

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


def process_trade(msg) -> None:
    trace_id = ""
    span_id = ""
    try:
        trace_id = msg.traceId
    except Exception:
        pass
    try:
        comment = msg.comment
        if comment.startswith("span="):
            parts = comment.split(";", 1)
            span_id = parts[0][5:]
            comment = parts[1] if len(parts) > 1 else ""
        record = {
            "event_id": msg.eventId,
            "event_time": msg.eventTime,
            "broker_time": msg.brokerTime,
            "local_time": msg.localTime,
            "action": msg.action,
            "ticket": msg.ticket,
            "magic": msg.magic,
            "source": msg.source,
            "symbol": msg.symbol,
            "order_type": msg.orderType,
            "lots": msg.lots,
            "price": msg.price,
            "sl": msg.sl,
            "tp": msg.tp,
            "profit": msg.profit,
            "comment": comment,
            "remaining_lots": msg.remainingLots,
            "decision_id": msg.decisionId,
        }
        record = TradeEvent(**record).dict()
    except (AttributeError, ValidationError) as e:
        logger.warning({"error": "invalid trade event", "details": str(e)})
        return
    with tracer.start_as_current_span("process_event") as span:
        ctx = span.get_span_context()
        record.setdefault("trace_id", trace_id or format_trace_id(ctx.trace_id))
        record.setdefault("span_id", span_id or format_span_id(ctx.span_id))
        logger.info(record)
        append_csv(LOG_FILES[TRADE_MSG], record)
        if ws_trades:
            try:
                ws_trades.send(json.dumps(record))
            except Exception:
                pass


def process_metric(msg) -> None:
    try:
        record = {
            "time": msg.time,
            "magic": msg.magic,
            "win_rate": msg.winRate,
            "avg_profit": msg.avgProfit,
            "trade_count": msg.tradeCount,
            "drawdown": msg.drawdown,
            "sharpe": msg.sharpe,
            "file_write_errors": msg.fileWriteErrors,
            "socket_errors": msg.socketErrors,
            "book_refresh_seconds": msg.bookRefreshSeconds,
        }
        record = MetricEvent(**record).dict()
    except (AttributeError, ValidationError) as e:
        logger.warning({"error": "invalid metric event", "details": str(e)})
        return
    with tracer.start_as_current_span("process_metric") as span:
        ctx = span.get_span_context()
        record.setdefault("trace_id", format_trace_id(ctx.trace_id))
        record["span_id"] = format_span_id(ctx.span_id)
        logger.info(record)
        append_csv(LOG_FILES[METRIC_MSG], record)
        if ws_metrics:
            try:
                ws_metrics.send(json.dumps(record))
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description="Listen for observer events over ZeroMQ")
    p.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5556",
        help="ZeroMQ PUB endpoint to connect to",
    )
    p.add_argument("--ws-url", help="Dashboard websocket base URL, e.g. ws://localhost:8000", default="")
    p.add_argument("--api-token", default=os.getenv("DASHBOARD_API_TOKEN", ""), help="API token for dashboard authentication")
    args = p.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    import time
    time.sleep(0.2)

    if args.ws_url and create_connection:
        global ws_trades, ws_metrics
        try:
            ws_trades = create_connection(f"{args.ws_url}/ws/trades?token={args.api_token}")
            ws_metrics = create_connection(f"{args.ws_url}/ws/metrics?token={args.api_token}")
        except Exception:
            ws_trades = None
            ws_metrics = None

    write_run_info()
    for p in LOG_FILES.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open("w") as f:
                f.write("symbol\nX\n")
    try:
        while True:
            data = sub.recv()
            if not data:
                continue
            kind = data[0]
            payload = data[1:]
            if kind == TRADE_MSG:
                with trade_capnp.TradeEvent.from_bytes(payload) as msg:
                    process_trade(msg)
            elif kind == METRIC_MSG:
                with metrics_capnp.Metrics.from_bytes(payload) as msg:
                    process_metric(msg)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        ctx.term()
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
