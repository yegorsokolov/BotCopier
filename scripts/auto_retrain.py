#!/usr/bin/env python3
"""Automatically retrain when live metrics degrade."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

import logging
import os
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
from opentelemetry.trace import format_trace_id, format_span_id

from scripts.train_target_clone import train as train_model
from scripts.backtest_strategy import run_backtest
from scripts.publish_model import publish

STATE_FILE = "last_event_id"

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "auto_retrain")})
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


def _read_last_event_id(out_dir: Path) -> int:
    """Return the last processed event id."""
    try:
        return int((out_dir / STATE_FILE).read_text().strip())
    except Exception:
        return 0


def _write_last_event_id(out_dir: Path, event_id: int) -> None:
    (out_dir / STATE_FILE).write_text(str(int(event_id)))


def _load_latest_metrics(metrics_file: Path) -> Optional[Dict[str, float]]:
    if not metrics_file.exists():
        return None
    last: Optional[Dict[str, str]] = None
    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            last = row
    if not last:
        return None
    try:
        return {
            "win_rate": float(last.get("win_rate", 0) or 0),
            "drawdown": float(last.get("drawdown", 0) or 0),
        }
    except Exception:
        return None


def retrain_if_needed(
    log_dir: Path,
    out_dir: Path,
    files_dir: Path,
    *,
    metrics_file: Optional[Path] = None,
    win_rate_threshold: float = 0.4,
    drawdown_threshold: float = 0.2,
    tick_file: Optional[Path] = None,
) -> bool:
    """Retrain and publish a model when win rate or drawdown worsens."""
    with tracer.start_as_current_span("retrain_if_needed"):
        metrics_path = metrics_file or (log_dir / "metrics.csv")
        metrics = _load_latest_metrics(metrics_path)
        if not metrics:
            logger.info("no metrics available")
            return False

        if metrics["win_rate"] >= win_rate_threshold and metrics["drawdown"] <= drawdown_threshold:
            logger.info("metrics within thresholds")
            return False

        logger.info("retraining model")
        # Load last processed event id for incremental training
        import scripts.train_target_clone as tc

        last_id = _read_last_event_id(out_dir)
        tc.START_EVENT_ID = last_id
        train_model(log_dir, out_dir, incremental=True)

        model_json = out_dir / "model.json"
        model_onnx = out_dir / "model.onnx"
        try:
            data = json.loads(model_json.read_text())
            _write_last_event_id(out_dir, int(data.get("last_event_id", last_id)))
        except Exception:
            pass

        # Backtest to ensure new model improves over previous metrics
        backtest_file = tick_file or (log_dir / "trades_raw.csv")
        try:
            result = run_backtest(model_json, backtest_file)
        except Exception:
            return False

        if result.get("win_rate", 0) <= metrics["win_rate"] or result.get("drawdown", 1) >= metrics["drawdown"]:
            logger.info("backtest did not improve metrics")
            return False

        publish(model_onnx if model_onnx.exists() else model_json, files_dir)
        tc.START_EVENT_ID = 0
        logger.info("retrain complete")
        return True


def main() -> None:
    span = tracer.start_span("auto_retrain")
    ctx = span.get_span_context()
    logger.info("auto retrain start", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})
    p = argparse.ArgumentParser(description="Retrain model when metrics degrade")
    p.add_argument("--log-dir", required=True, help="directory with observer logs")
    p.add_argument("--out-dir", required=True, help="output model directory")
    p.add_argument("--files-dir", required=True, help="MT4 Files directory")
    p.add_argument("--metrics-file", help="path to metrics.csv")
    p.add_argument("--tick-file", help="tick or trades file for backtesting")
    p.add_argument("--win-rate-threshold", type=float, default=0.4)
    p.add_argument("--drawdown-threshold", type=float, default=0.2)
    p.add_argument("--interval", type=float, help="seconds between checks (loop)" )
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    files_dir = Path(args.files_dir)
    metrics_path = Path(args.metrics_file) if args.metrics_file else None
    tick_path = Path(args.tick_file) if args.tick_file else None

    if args.interval:
        while True:
            retrain_if_needed(
                log_dir,
                out_dir,
                files_dir,
                metrics_file=metrics_path,
                win_rate_threshold=args.win_rate_threshold,
                drawdown_threshold=args.drawdown_threshold,
                tick_file=tick_path,
            )
            time.sleep(args.interval)
    else:
        retrain_if_needed(
            log_dir,
            out_dir,
            files_dir,
            metrics_file=metrics_path,
            win_rate_threshold=args.win_rate_threshold,
            drawdown_threshold=args.drawdown_threshold,
            tick_file=tick_path,
        )
    logger.info("auto retrain finished", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})
    span.end()


if __name__ == "__main__":
    main()
