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
try:
    import uvloop
    uvloop.install()
except Exception:
    pass
import time
import pickle
import subprocess
import gzip
import threading
from datetime import datetime
import psutil
try:  # optional drift detection dependency
    from river.drift import ADWIN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ADWIN = None  # type: ignore
try:  # optional NATS dependency
    import nats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nats = None
try:  # pydantic validation schemas
    from pydantic import ValidationError  # type: ignore
    from schemas.trades import TradeEvent  # type: ignore
    from schemas.metrics import MetricEvent  # type: ignore
except Exception:  # pragma: no cover - minimal fallback
    class ValidationError(Exception):
        pass

    class _Event(dict):
        def __init__(self, **data):
            super().__init__(data)

        def dict(self):  # mimic pydantic API
            return dict(self)

    TradeEvent = MetricEvent = _Event  # type: ignore
try:  # optional websocket client
    from websocket import create_connection, WebSocket  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    create_connection = None
    WebSocket = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # optional OpenTelemetry dependencies
    from opentelemetry import trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    try:
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
except Exception:  # pragma: no cover - minimal fallbacks
    trace = None  # type: ignore
    set_logger_provider = lambda *a, **k: None  # type: ignore
    OTLPSpanExporter = OTLPLogExporter = JaegerExporter = None  # type: ignore

    class Resource:  # type: ignore
        @staticmethod
        def create(*a, **k):
            return None

    class TracerProvider:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def add_span_processor(self, *a, **k):
            pass

    class LoggerProvider:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def add_log_record_processor(self, *a, **k):
            pass

    class BatchSpanProcessor:  # type: ignore
        def __init__(self, *a, **k):
            pass

    class BatchLogRecordProcessor:  # type: ignore
        def __init__(self, *a, **k):
            pass

    class LoggingHandler(logging.Handler):  # type: ignore
        def __init__(self, *a, **k):
            super().__init__()

        def emit(self, record):
            pass

    def format_span_id(x):  # type: ignore
        return str(x)

    def format_trace_id(x):  # type: ignore
        return str(x)

    class NonRecordingSpan:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

    class SpanContext:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

    class TraceFlags(int):  # type: ignore
        SAMPLED = 1

    class TraceState:  # type: ignore
        pass

    def set_span_in_context(span):  # type: ignore
        return None

    class _SpanCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        class _Ctx:
            trace_id = 0
            span_id = 0

        def get_span_context(self):  # type: ignore
            return self._Ctx()

    class _TracerStub:
        def start_as_current_span(self, *a, **k):
            return _SpanCtx()

    class _TraceStub:
        def set_tracer_provider(self, *a, **k):
            pass

        def get_tracer(self, *a, **k):
            return _TracerStub()

    trace = _TraceStub()

try:  # Cap'n Proto schemas loaded at runtime
    from proto import trade_capnp, metrics_capnp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trade_capnp = metrics_capnp = None  # type: ignore

try:
    from .shm_ring import ShmRing, TRADE_MSG, METRIC_MSG
except Exception:  # pragma: no cover - package relative import fallback
    from shm_ring import ShmRing, TRADE_MSG, METRIC_MSG

SCHEMA_VERSION = 1
TRADE_MSG = 0
METRIC_MSG = 1
HELLO_MSG = 2

LOG_FILES = {
    TRADE_MSG: Path("logs/trades_raw.csv"),
    METRIC_MSG: Path("logs/metrics.csv"),
}

# system telemetry log
TELEMETRY_LOG = Path("logs/system_telemetry.csv")

current_trace_id = ""
current_span_id = ""

RUN_INFO_PATH = Path("logs/run_info.json")
run_info_written = False

trade_prev = bytearray()
metric_prev = bytearray()

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

# ADWIN drift detectors
ADWIN_STATE_PATH = Path("logs/adwin_state.pkl")
MONITOR_FEATURES = ["win_rate", "avg_profit", "drawdown", "sharpe"]
adwin_detectors: dict[str, ADWIN] = {}

def _load_adwin() -> None:
    if ADWIN is None:
        return
    global adwin_detectors
    if ADWIN_STATE_PATH.exists():
        try:
            with ADWIN_STATE_PATH.open("rb") as f:
                adwin_detectors = pickle.load(f)
        except Exception:
            adwin_detectors = {}
    for feat in MONITOR_FEATURES:
        adwin_detectors.setdefault(feat, ADWIN())

def _save_adwin() -> None:
    if ADWIN is None:
        return
    try:
        ADWIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ADWIN_STATE_PATH.open("wb") as f:
            pickle.dump(adwin_detectors, f)
    except Exception:
        pass

def _trigger_retrain() -> None:
    try:
        subprocess.Popen([sys.executable, str(Path(__file__).with_name("auto_retrain.py"))])
    except Exception as e:
        logger.error({"error": "failed to trigger retrain", "details": str(e)})

_load_adwin()


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


def capture_system_metrics() -> None:
    """Capture CPU, memory and network stats with the latest trace id."""
    global current_trace_id, current_span_id
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    net = psutil.net_io_counters()
    record = {
        "time": datetime.utcnow().isoformat(),
        "cpu_percent": cpu,
        "mem_percent": mem,
        "bytes_sent": net.bytes_sent,
        "bytes_recv": net.bytes_recv,
        "trace_id": current_trace_id,
        "span_id": current_span_id,
    }
    append_csv(TELEMETRY_LOG, record)


def _system_monitor(interval: float = 5.0) -> None:
    while True:
        try:
            capture_system_metrics()
        except Exception:
            pass
        time.sleep(interval)


def start_system_monitor(interval: float = 5.0) -> None:
    t = threading.Thread(target=_system_monitor, args=(interval,), daemon=True)
    t.start()


def _maybe_decompress(data: bytes) -> bytes:
    if len(data) > 2 and data[0] == 0x1F and data[1] == 0x8B:
        try:
            return gzip.decompress(data)
        except Exception:
            return data
    return data


def _decode_event(data: bytes, prev: bytearray):
    buf = _maybe_decompress(data)
    if len(buf) < 2:
        return None
    version = buf[0]
    tag = buf[1]
    if tag == 0:
        prev[:] = buf[2:]
        return version, bytes(prev)
    if tag == 1:
        if not prev:
            return None
        cnt = buf[2]
        pos = 3
        for _ in range(cnt):
            if pos + 2 >= len(buf):
                break
            idx = (buf[pos] << 8) | buf[pos + 1]
            val = buf[pos + 2]
            pos += 3
            if idx < len(prev):
                prev[idx] = val
        return version, bytes(prev)
    return None


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
    global current_trace_id, current_span_id
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
        current_trace_id = record.get("trace_id", "")
        current_span_id = record.get("span_id", "")
        if ws_trades:
            try:
                ws_trades.send(json.dumps(record))
            except Exception:
                pass


def process_metric(msg) -> None:
    global current_trace_id, current_span_id
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
        current_trace_id = record.get("trace_id", "")
        current_span_id = record.get("span_id", "")
        if ws_metrics:
            try:
                ws_metrics.send(json.dumps(record))
            except Exception:
                pass
        if ADWIN is not None:
            drift_features = []
            for feat in MONITOR_FEATURES:
                try:
                    val = float(record.get(feat, 0))
                except (TypeError, ValueError):
                    continue
                det = adwin_detectors.get(feat)
                if det is None:
                    continue
                det.update(val)
                if det.drift_detected:
                    drift_features.append(feat)
            if drift_features:
                logger.warning(
                    {"alert": "drift detected", "features": drift_features},
                    extra=extra,
                )
                _trigger_retrain()
            _save_adwin()


def main() -> int:
    p = argparse.ArgumentParser(description="Listen for observer events")
    p.add_argument("--servers", default="nats://127.0.0.1:4222", help="NATS server URLs")
    p.add_argument("--ws-url", help="Dashboard websocket base URL, e.g. ws://localhost:8000", default="")
    p.add_argument("--api-token", default=os.getenv("DASHBOARD_API_TOKEN", ""), help="API token for dashboard authentication")
    p.add_argument("--ring-path", default=os.getenv("TBOT_RING", "/tmp/tbot_events"), help="Path to shared memory ring buffer")
    args = p.parse_args()

    start_system_monitor()

    if args.ws_url and create_connection:
        global ws_trades, ws_metrics
        try:
            headers = [f"Authorization: Bearer {args.api_token}"] if args.api_token else []
            ws_trades = create_connection(f"{args.ws_url}/ws/trades", header=headers)
            ws_metrics = create_connection(f"{args.ws_url}/ws/metrics", header=headers)
        except Exception:
            ws_trades = None
            ws_metrics = None

    # Try shared memory ring first
    ring = None
    try:
        ring = ShmRing.open(args.ring_path)
    except Exception:
        ring = None

    if ring is not None:
        while True:
            msg = ring.pop()
            if msg is None:
                time.sleep(0.01)
                continue
            msg_type, payload = msg
            if msg_type != HELLO_MSG:
                logger.warning("expected hello packet")
                return 1
            try:
                info = json.loads(payload.tobytes().decode())
            except Exception:
                logger.warning("invalid hello packet")
                return 1
            version = info.get("schema_version")
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %s mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                return 1
            break
        write_run_info()
        for pth in LOG_FILES.values():
            pth.parent.mkdir(parents=True, exist_ok=True)
        while True:
            msg = ring.pop()
            if msg is None:
                time.sleep(0.01)
                continue
            msg_type, payload = msg
            decoded = _decode_event(bytes(payload), trade_prev if msg_type == TRADE_MSG else metric_prev)
            if not decoded:
                continue
            version, body = decoded
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %d mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                continue
            try:
                if msg_type == TRADE_MSG:
                    trade = trade_capnp.TradeEvent.from_bytes(body)
                    process_trade(trade)
                elif msg_type == METRIC_MSG:
                    metric = metrics_capnp.Metrics.from_bytes(body)
                    process_metric(metric)
            except Exception:
                continue
        return 0

    async def _run() -> None:
        nc = await nats.connect(args.servers)
        js = nc.jetstream()

        async def trade_handler(msg):
            decoded = _decode_event(msg.data, trade_prev)
            if not decoded:
                await msg.ack()
                return
            version, body = decoded
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %d mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                await msg.ack()
                return
            try:
                trade = trade_capnp.TradeEvent.from_bytes(body)
            except Exception:
                try:
                    trade = trade_capnp.TradeEvent.from_bytes_packed(body)
                except Exception:
                    await msg.ack()
                    return
            process_trade(trade)
            await msg.ack()

        async def metric_handler(msg):
            decoded = _decode_event(msg.data, metric_prev)
            if not decoded:
                await msg.ack()
                return
            version, body = decoded
            if version != SCHEMA_VERSION:
                logger.warning(
                    "schema version %d mismatch (expected %d)",
                    version,
                    SCHEMA_VERSION,
                )
                await msg.ack()
                return
            try:
                metric = metrics_capnp.Metrics.from_bytes(body)
            except Exception:
                try:
                    metric = metrics_capnp.Metrics.from_bytes_packed(body)
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
