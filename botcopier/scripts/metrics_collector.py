#!/usr/bin/env python3
"""Listen for metric messages via Arrow Flight and store them in a SQLite database."""

try:
    import uvloop

    uvloop.install()
except Exception:
    pass

import argparse
import asyncio
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections import deque
from contextlib import closing, suppress
from datetime import datetime
from pathlib import Path

import pandas as pd

try:  # prefer systemd journal if available
    from systemd.journal import JournalHandler

    logging.basicConfig(handlers=[JournalHandler()], level=logging.INFO)
except Exception:  # pragma: no cover - fallback to file logging
    logging.basicConfig(filename="metrics_collector.log", level=logging.INFO)
from asyncio import Queue
from typing import Callable, Optional

from aiohttp import web

try:
    from river.drift import ADWIN
except Exception:  # pragma: no cover - optional dependency

    class ADWIN:  # minimal stub
        def update(self, x):
            return False

        @property
        def estimation(self):
            return 0.0


import psutil

try:  # optional systemd notification support
    from systemd import daemon
except Exception:  # pragma: no cover - systemd not installed
    daemon = None

from google.protobuf.json_format import MessageToDict

from proto import metric_event_pb2

try:  # optional Arrow Flight dependency
    import pyarrow.flight as flight  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    flight = None  # type: ignore

SCHEMA_VERSION = 1

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider

try:  # optional OTLP exporters
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - exporters not installed
    OTLPSpanExporter = OTLPLogExporter = OTLPMetricExporter = None  # type: ignore
from opentelemetry.metrics import Observation
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
    format_span_id,
    format_trace_id,
    set_span_in_context,
)

if __package__:
    from ..features import indicator_discovery, technical
else:  # pragma: no cover - executed when run as script
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from botcopier.features import indicator_discovery, technical

FIELDS = [
    "time",
    "magic",
    "win_rate",
    "avg_profit",
    "trade_count",
    "drawdown",
    "sharpe",
    "sortino",
    "expectancy",
    "cvar",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "file_write_errors",
    "socket_errors",
    "cpu_load",
    "flush_latency_ms",
    "network_latency_ms",
    "book_refresh_seconds",
    "var_breach_count",
    "trade_queue_depth",
    "metric_queue_depth",
    "trade_retry_count",  # backpressure on trade sends
    "metric_retry_count",  # backpressure on metric sends
    "fallback_events",
    "risk_weight",
    "trace_id",
    "span_id",
    "queue_backlog",
]

resource = Resource.create(
    {"service.name": os.getenv("OTEL_SERVICE_NAME", "metrics_collector")}
)
provider = TracerProvider(resource=resource)
if (
    endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
) and OTLPSpanExporter is not None:
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

metric_readers = []
if endpoint and OTLPMetricExporter is not None:
    metric_readers.append(
        PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
    )
meter_provider = MeterProvider(metric_readers=metric_readers, resource=resource)
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)


def _sd_notify_ready() -> None:
    if daemon is not None:
        daemon.sd_notify("READY=1")


async def _watchdog_task(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        if daemon is not None:
            daemon.sd_notify("WATCHDOG=1")


def _cpu_usage(_):
    return [Observation(psutil.cpu_percent(interval=None))]


def _mem_usage(_):
    return [Observation(psutil.virtual_memory().percent)]


def _net_io(_):
    net = psutil.net_io_counters()
    return [
        Observation(net.bytes_sent, {"direction": "sent"}),
        Observation(net.bytes_recv, {"direction": "recv"}),
    ]


meter.create_observable_gauge(
    "system.cpu.percent",
    [_cpu_usage],
    unit="percent",
    description="CPU usage percentage",
)
meter.create_observable_gauge(
    "system.memory.percent",
    [_mem_usage],
    unit="percent",
    description="Memory usage percentage",
)
meter.create_observable_gauge(
    "system.network.bytes",
    [_net_io],
    unit="bytes",
    description="Network I/O in bytes",
)


def _context_from_ids(trace_id: str, span_id: str):
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


logger_provider = LoggerProvider(resource=resource)
if endpoint and OTLPLogExporter is not None:
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
    )
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


def _update_model(
    model_json: Path,
    metric: float,
    retrained: bool,
    indicators: list[str] | None = None,
) -> None:
    """Record drift metrics, retrain timestamps and indicator history."""
    ts = datetime.utcnow().isoformat()
    data: dict[str, object] = {}
    if model_json.exists():
        try:
            data = json.loads(model_json.read_text())
        except Exception:
            data = {}
    data["drift_metric"] = float(metric)
    data["drift_metrics"] = {"adwin": float(metric)}
    history = data.setdefault("drift_history", [])
    if isinstance(history, list):
        history.append({"time": ts, "adwin": float(metric)})
    if retrained:
        retrain_hist = data.setdefault("retrain_history", [])
        if isinstance(retrain_hist, list):
            retrain_hist.append({"time": ts, "adwin": float(metric)})
    if indicators:
        ind_hist = data.setdefault("indicator_history", [])
        if isinstance(ind_hist, list):
            ind_hist.append({"time": ts, "formulas": indicators})
    model_json.write_text(json.dumps(data, indent=2))


def _run_indicator_script(
    script: Path, df: pd.DataFrame, model_json: Path
) -> list[str] | None:
    """Run an external indicator generator and return produced formulas."""

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = Path(tmp.name)
    cmd = [sys.executable, str(script), str(csv_path), "--model", str(model_json)]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError):
        logger.exception("indicator generation failed")
        return None
    finally:
        with suppress(FileNotFoundError):
            csv_path.unlink()

    technical.refresh_symbolic_indicators(model_json)
    try:
        data = json.loads(model_json.read_text())
        return data.get("symbolic_indicators", {}).get("formulas")
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load generated symbolic indicators")
        return None


async def _writer_task(
    db_file: Path,
    queue: Queue,
    prom_updater: Callable[[dict], None] | None = None,
    *,
    drift_metric: str | None = None,
    drift_threshold: float = 0.0,
    model_json: Path | None = None,
    log_dir: Path | None = None,
    out_dir: Path | None = None,
    files_dir: Path | None = None,
    detector: ADWIN | None = None,
    indicator_script: Path | None = None,
    neural_generator: Path | None = None,
) -> None:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    cols = ",".join(FIELDS)
    placeholders = ",".join(["?"] * len(FIELDS))
    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS metrics ({','.join([f'{c} TEXT' for c in FIELDS])})"
        )
        insert_sql = f"INSERT INTO metrics ({cols}) VALUES ({placeholders})"
        prev_est: float | None = None
        recent: deque[dict] = deque(maxlen=100)
        while True:
            row = await queue.get()
            if row is None:
                break
            ctx_in = _context_from_ids(row.get("trace_id", ""), row.get("span_id", ""))
            with tracer.start_as_current_span("metrics_store", context=ctx_in) as span:
                span.set_attribute("file_write_errors", row.get("file_write_errors", 0))
                span.set_attribute("socket_errors", row.get("socket_errors", 0))
                span.set_attribute("fallback_events", row.get("fallback_events", 0))
                span.set_attribute("queue_backlog", row.get("queue_backlog", 0))
                try:
                    conn.execute(insert_sql, [row.get(f, "") for f in FIELDS])
                    conn.commit()
                except (sqlite3.Error, OSError) as exc:  # pragma: no cover - disk or schema issues
                    logger.exception(
                        {"error": "file write failure", "details": str(exc)}
                    )
                else:
                    recent.append(row)
                    if prom_updater is not None:
                        prom_updater(row)
                    if (
                        detector is not None
                        and drift_metric
                        and model_json is not None
                        and (val := row.get(drift_metric)) is not None
                    ):
                        try:
                            val_f = float(val)
                        except (TypeError, ValueError):
                            pass
                        else:
                            changed = detector.update(val_f)
                            if prev_est is None:
                                prev_est = detector.estimation
                            if changed:
                                diff = abs((prev_est or 0.0) - detector.estimation)
                                retrain = diff > drift_threshold
                                formulas: list[str] | None = None
                                if recent and retrain:
                                    df = pd.DataFrame(list(recent))
                                    cols_num = [
                                        c
                                        for c in df.columns
                                        if c
                                        not in {"time", "magic", "trace_id", "span_id"}
                                    ]
                                    if cols_num:
                                        df = (
                                            df[cols_num]
                                            .apply(pd.to_numeric, errors="coerce")
                                            .fillna(0.0)
                                        )
                                if indicator_script or neural_generator:
                                    script = indicator_script or neural_generator
                                    formulas = await asyncio.to_thread(
                                        _run_indicator_script,
                                        script,
                                        df,
                                        model_json,
                                    )
                                    if formulas:
                                        technical.append_symbolic_indicators(
                                            formulas, cols_num, model_json
                                        )
                                else:
                                    formulas = indicator_discovery.evolve_indicators(
                                        df, cols_num, model_path=model_json
                                    )
                                    technical.refresh_symbolic_indicators(model_json)
                                _update_model(model_json, diff, retrain, formulas)
                                if retrain and log_dir and out_dir and files_dir:
                                    base = Path(__file__).resolve().parent
                                    try:
                                        await asyncio.to_thread(
                                            subprocess.run,
                                            [
                                                sys.executable,
                                                str(base / "train_target_clone.py"),
                                                str(log_dir),
                                                str(out_dir),
                                            ],
                                            check=True,
                                        )
                                        await asyncio.to_thread(
                                            subprocess.run,
                                            [
                                                sys.executable,
                                                str(
                                                    base / "generate_mql4_from_model.py"
                                                ),
                                                "--model",
                                                str(model_json),
                                                "--template",
                                                str(
                                                    base.parent / "StrategyTemplate.mq4"
                                                ),
                                            ],
                                            check=True,
                                        )
                                    except Exception:
                                        logger.exception("retrain failed")
                            prev_est = detector.estimation
            queue.task_done()


def serve(
    db_file: Path,
    http_host: str = "127.0.0.1",
    http_port: Optional[int] = None,
    prom_port: Optional[int] = None,
    flight_host: str = "127.0.0.1",
    flight_port: int = 8815,
    *,
    drift_metric: str | None = None,
    drift_threshold: float = 0.05,
    model_json: Path = Path("model.json"),
    log_dir: Path | None = None,
    out_dir: Path | None = None,
    files_dir: Path | None = None,
) -> None:
    async def _run() -> None:
        queue: Queue = Queue()
        prom_updater: Callable[[dict], None]
        detector = ADWIN() if drift_metric else None
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        watchdog_interval = None
        if daemon is not None:
            try:
                watchdog_interval = int(os.getenv("WATCHDOG_USEC", "0")) / 2_000_000
            except ValueError:
                watchdog_interval = None
            if watchdog_interval:
                asyncio.create_task(_watchdog_task(watchdog_interval))

        prom_runner = None
        if prom_port is not None:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                Counter,
                Gauge,
                generate_latest,
            )

            prom_app = web.Application()

            async def prom_handler(_request: web.Request) -> web.Response:
                return web.Response(
                    body=generate_latest(), content_type=CONTENT_TYPE_LATEST
                )

            prom_app.add_routes([web.get("/metrics", prom_handler)])
            prom_runner = web.AppRunner(prom_app)
            await prom_runner.setup()
            prom_site = web.TCPSite(prom_runner, http_host, prom_port)
            await prom_site.start()
            win_rate_g = Gauge("bot_win_rate", "Win rate")
            drawdown_g = Gauge("bot_drawdown", "Drawdown")
            cvar_g = Gauge("bot_cvar", "Conditional Value at Risk (worst 5%)")
            socket_err_c = Counter("bot_socket_errors_total", "Socket error count")
            file_err_c = Counter(
                "bot_file_write_errors_total", "File write error count"
            )
            fallback_event_c = Counter(
                "bot_fallback_events_total",
                "Fallback logging event count",
            )
            cpu_load_g = Gauge("bot_cpu_load", "CPU load")
            flush_latency_g = Gauge("bot_flush_latency_ms", "Log flush latency in ms")
            network_latency_g = Gauge(
                "bot_network_latency_ms", "Network send latency in ms"
            )
            book_refresh_g = Gauge(
                "bot_book_refresh_seconds",
                "Cached book refresh interval",
            )
            host_cpu_g = Gauge("system_cpu_percent", "Host CPU utilisation")
            host_mem_g = Gauge("system_memory_percent", "Host memory utilisation")
            metric_queue_g = Gauge(
                "bot_metric_queue_depth", "Arrow Flight metric queue depth"
            )
            trade_queue_g = Gauge(
                "bot_trade_queue_depth", "Arrow Flight trade queue depth"
            )
            wal_size_g = Gauge("bot_wal_bytes", "Total WAL size in bytes")
            trade_retry_g = Gauge("bot_trade_retry_count", "Trade send retry count")
            metric_retry_g = Gauge("bot_metric_retry_count", "Metric send retry count")
            queue_backlog_g = Gauge(
                "bot_queue_backlog", "Observer fallback queue backlog"
            )
            wal_dir = Path(
                os.getenv(
                    "BOTCOPIER_LOG_DIR",
                    Path.home() / ".local/share/botcopier/logs",
                )
            )

            TRADE_Q_ALERT = 100
            METRIC_Q_ALERT = 100
            RETRY_ALERT = 5  # warn if retries exceed this threshold
            WAL_SIZE_ALERT = 1024 * 1024  # 1MB

            async def _sample_host_metrics() -> None:
                while True:
                    try:
                        host_cpu_g.set(psutil.cpu_percent(interval=None))
                        host_mem_g.set(psutil.virtual_memory().percent)
                    except Exception:
                        pass
                    await asyncio.sleep(5)

            asyncio.create_task(_sample_host_metrics())

            async def _sample_wal_metrics() -> None:
                while True:
                    try:
                        size = 0
                        for fn in ("pending_trades.wal", "pending_metrics.wal"):
                            fp = wal_dir / fn
                            if fp.exists():
                                size += fp.stat().st_size
                        wal_size_g.set(size)
                        if size > WAL_SIZE_ALERT:
                            logger.warning({"alert": "wal backlog", "wal_bytes": size})
                    except Exception:
                        pass
                    await asyncio.sleep(5)

            asyncio.create_task(_sample_wal_metrics())

            def _prom_updater(row: dict) -> None:
                if (v := row.get("win_rate")) is not None:
                    try:
                        win_rate_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("drawdown")) is not None:
                    try:
                        drawdown_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("cvar")) is not None:
                    try:
                        cvar_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("socket_errors")) is not None:
                    try:
                        socket_err_c.inc(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("file_write_errors")) is not None:
                    try:
                        file_err_c.inc(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("cpu_load")) is not None:
                    try:
                        cpu_load_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("flush_latency_ms")) is not None:
                    try:
                        flush_latency_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("network_latency_ms")) is not None:
                    try:
                        network_latency_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("book_refresh_seconds")) is not None:
                    try:
                        book_refresh_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("metric_queue_depth")) is not None:
                    try:
                        val = float(v)
                        metric_queue_g.set(val)
                        if val > METRIC_Q_ALERT:
                            logger.warning(
                                {"alert": "metric backlog", "metric_queue_depth": val}
                            )
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("trade_queue_depth")) is not None:
                    try:
                        val = float(v)
                        trade_queue_g.set(val)
                        if val > TRADE_Q_ALERT:
                            logger.warning(
                                {"alert": "trade backlog", "trade_queue_depth": val}
                            )
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("queue_backlog")) is not None:
                    try:
                        queue_backlog_g.set(float(v))
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("trade_retry_count")) is not None:
                    try:
                        val = float(v)
                        trade_retry_g.set(val)
                        if val > RETRY_ALERT:
                            logger.warning(
                                {"alert": "trade retries", "trade_retry_count": val}
                            )
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("metric_retry_count")) is not None:
                    try:
                        val = float(v)
                        metric_retry_g.set(val)
                        if val > RETRY_ALERT:
                            logger.warning(
                                {"alert": "metric retries", "metric_retry_count": val}
                            )
                    except (TypeError, ValueError):
                        pass
                if (v := row.get("fallback_events")) is not None:
                    try:
                        fallback_event_c.inc(float(v))
                    except (TypeError, ValueError):
                        pass

            prom_updater = _prom_updater
        else:
            prom_updater = lambda _row: None

        writer_task = asyncio.create_task(
            _writer_task(
                db_file,
                queue,
                prom_updater,
                drift_metric=drift_metric,
                drift_threshold=drift_threshold,
                model_json=model_json,
                log_dir=log_dir,
                out_dir=out_dir,
                files_dir=files_dir,
                detector=detector,
            )
        )
        _sd_notify_ready()

        if flight is None:
            raise RuntimeError("pyarrow.flight is required")
        client = flight.FlightClient(f"grpc://{flight_host}:{flight_port}")

        async def poll() -> None:
            offset = 0
            ticket = flight.Ticket(b"metrics")
            while True:
                try:
                    reader = client.do_get(ticket)
                    table = reader.read_all()
                    rows = table.to_pylist()
                    for row in rows[offset:]:
                        trace_id = row.get("trace_id", "")
                        span_id = row.get("span_id", "")
                        ctx_in = _context_from_ids(trace_id, span_id)
                        with tracer.start_as_current_span(
                            "metrics_message", context=ctx_in
                        ) as span:
                            ctx = span.get_span_context()
                            row.setdefault(
                                "trace_id", trace_id or format_trace_id(ctx.trace_id)
                            )
                            row.setdefault(
                                "span_id", span_id or format_span_id(ctx.span_id)
                            )
                            span.set_attribute(
                                "file_write_errors", row.get("file_write_errors", 0)
                            )
                            span.set_attribute(
                                "socket_errors", row.get("socket_errors", 0)
                            )
                            span.set_attribute(
                                "fallback_events", row.get("fallback_events", 0)
                            )
                            span.set_attribute(
                                "queue_backlog", row.get("queue_backlog", 0)
                            )
                            extra = {}
                            try:
                                extra["trace_id"] = int(row["trace_id"], 16)
                                extra["span_id"] = int(row["span_id"], 16)
                            except (KeyError, ValueError):
                                pass
                            logger.info(row, extra=extra)
                            await queue.put(row)
                    offset = len(rows)
                except Exception as e:  # pragma: no cover - network issues
                    logger.warning({"error": "flight error", "details": str(e)})
                    await asyncio.sleep(1)
                    continue
                await asyncio.sleep(1)

        poll_task = asyncio.create_task(poll())

        async def history_handler(request: web.Request) -> web.Response:
            limit_param = request.query.get("limit", "100")
            try:
                limit = int(limit_param)
            except ValueError:
                limit = 100
            with sqlite3.connect(db_file) as conn:
                rows = conn.execute(
                    "SELECT * FROM metrics ORDER BY ROWID DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            rows = [dict(zip(FIELDS, r)) for r in reversed(rows)]
            return web.json_response(rows)

        async def ingest_handler(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                return web.Response(status=400, text="invalid json")
            trace_id = data.get("trace_id", "")
            span_id = data.get("span_id", "")
            ctx_in = _context_from_ids(trace_id, span_id)
            with tracer.start_as_current_span(
                "metrics_http_ingest", context=ctx_in
            ) as span:
                ctx = span.get_span_context()
                data.setdefault("trace_id", trace_id or format_trace_id(ctx.trace_id))
                data.setdefault("span_id", span_id or format_span_id(ctx.span_id))
                span.set_attribute(
                    "file_write_errors", data.get("file_write_errors", 0)
                )
                span.set_attribute("socket_errors", data.get("socket_errors", 0))
                span.set_attribute("fallback_events", data.get("fallback_events", 0))
                span.set_attribute("queue_backlog", data.get("queue_backlog", 0))
                extra = {}
                try:
                    extra["trace_id"] = int(data["trace_id"], 16)
                    extra["span_id"] = int(data["span_id"], 16)
                except (KeyError, ValueError):
                    pass
                logger.info(data, extra=extra)
                await queue.put(data)
            return web.json_response({"status": "ok"})

        if http_port is not None:
            app = web.Application()
            app.add_routes(
                [
                    web.get("/history", history_handler),
                    web.post("/ingest", ingest_handler),
                ]
            )
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, http_host, http_port)
            await site.start()
        else:
            runner = None

        await stop.wait()

        poll_task.cancel()
        with suppress(asyncio.CancelledError):
            await poll_task
        if runner is not None:
            await runner.cleanup()
        if prom_runner is not None:
            await prom_runner.cleanup()
        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser(description="Collect metric messages into SQLite")
    p.add_argument(
        "--flight-host", default="127.0.0.1", help="Arrow Flight server host"
    )
    p.add_argument("--flight-port", type=int, default=8815)
    p.add_argument("--db", required=True, help="output SQLite file")
    p.add_argument("--http-host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, help="serve metrics via HTTP on this port")
    p.add_argument(
        "--prom-port",
        type=int,
        help="expose Prometheus metrics on this port",
    )
    p.add_argument("--drift-metric", help="metric field to monitor with ADWIN")
    p.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="minimum mean shift to trigger retrain",
    )
    p.add_argument(
        "--model-json",
        type=Path,
        default=Path("model.json"),
        help="path to model.json for audit updates",
    )
    p.add_argument("--log-dir", type=Path, help="directory with training logs")
    p.add_argument("--out-dir", type=Path, help="output directory for model artifacts")
    p.add_argument("--files-dir", type=Path, help="MT4 Files directory")
    args = p.parse_args()

    serve(
        Path(args.db),
        args.http_host,
        args.http_port,
        args.prom_port,
        args.flight_host,
        args.flight_port,
        drift_metric=args.drift_metric,
        drift_threshold=args.drift_threshold,
        model_json=args.model_json,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        files_dir=args.files_dir,
    )


if __name__ == "__main__":
    main()
