#!/usr/bin/env python3
"""Incrementally update a model from streaming trade events.

The trainer is designed to run continuously.  It tails
``logs/trades_raw.csv`` or consumes newline-delimited JSON records from a
socket.  After each batch it updates an :class:`~sklearn.linear_model.SGDClassifier`
using :meth:`partial_fit` and persists the coefficients to ``model.json``.
Whenever the coefficients change the script invokes
``generate_mql4_from_model.py`` so the corresponding Expert Advisor can be
reloaded by MetaTrader.

This module intentionally keeps dependencies light so it can run alongside the
MetaTrader terminal on modest hardware.  Before processing each batch the
trainer samples CPU usage with :func:`psutil.cpu_percent`.  If system load
exceeds 80% it sleeps briefly to yield the processor, adaptively throttling
consumption to avoid overwhelming the host.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
import time
import os
import threading
from pathlib import Path
from typing import Iterable, List, Dict, Any

try:  # prefer systemd journal if available
    from systemd.journal import JournalHandler
    logging.basicConfig(handlers=[JournalHandler()], level=logging.INFO)
except Exception:  # pragma: no cover - fallback to file logging
    logging.basicConfig(filename="online_trainer.log", level=logging.INFO)

import numpy as np
import psutil
from sklearn.linear_model import SGDClassifier

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "online_trainer")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
    )
set_logger_provider(logger_provider)
otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


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
otel_handler.setFormatter(JsonFormatter())
logger.addHandler(otel_handler)
logger.setLevel(logging.INFO)

try:  # optional systemd notification support
    from systemd import daemon
except Exception:  # pragma: no cover - systemd not installed
    daemon = None


def _sd_notify_ready() -> None:
    if daemon is not None:
        daemon.sd_notify("READY=1")


def _start_watchdog_thread() -> None:
    if daemon is None:
        return
    try:
        usec = int(os.getenv("WATCHDOG_USEC", "0"))
    except ValueError:
        usec = 0
    interval = usec / 2_000_000 if usec else 30
    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                daemon.sd_notify("WATCHDOG=1")
            except Exception:
                pass
    threading.Thread(target=_loop, daemon=True).start()


class OnlineTrainer:
    """Manage incremental updates and model persistence."""

    def __init__(
        self,
        model_path: Path | str = Path("model.json"),
        batch_size: int = 32,
        run_generator: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.run_generator = run_generator
        self.clf = SGDClassifier(loss="log_loss")
        self.feature_names: List[str] = []
        self._prev_coef: List[float] | None = None
        self.training_mode = "lite"
        if self.model_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Restore coefficients from ``model.json`` if present."""
        try:
            data = json.loads(self.model_path.read_text())
        except Exception:
            return
        self.training_mode = data.get("mode") or data.get("training_mode", "lite")
        self.feature_names = data.get("feature_names", [])
        coef = data.get("coefficients")
        intercept = data.get("intercept")
        if self.feature_names and coef is not None and intercept is not None:
            n = len(self.feature_names)
            self.clf.partial_fit(np.zeros((1, n)), [0], classes=np.array([0, 1]))
            self.clf.coef_ = np.array([coef])
            self.clf.intercept_ = np.array([intercept])
            self._prev_coef = list(coef) + [intercept]

    def _save(self) -> None:
        payload = {
            "feature_names": self.feature_names,
            "coefficients": self.clf.coef_[0].tolist(),
            "intercept": float(self.clf.intercept_[0]),
            "training_mode": self.training_mode,
            "mode": self.training_mode,
        }
        self.model_path.write_text(json.dumps(payload))

    def _maybe_generate(self) -> None:
        if not self.run_generator:
            return
        script = Path(__file__).resolve().with_name("generate_mql4_from_model.py")
        experts_dir = Path(__file__).resolve().parents[1] / "experts"
        cmd = [sys.executable, str(script), str(self.model_path), str(experts_dir)]
        if self.training_mode == "lite":
            cmd.append("--lite-mode")
        subprocess.run(cmd, check=False)

    # ------------------------------------------------------------------
    # Incremental training
    # ------------------------------------------------------------------
    def _ensure_features(self, keys: Iterable[str]) -> None:
        new_feats = [k for k in keys if k not in self.feature_names and k != "y"]
        if not new_feats:
            return
        self.feature_names.extend(sorted(new_feats))
        if hasattr(self.clf, "coef_"):
            n = len(self.feature_names)
            coef = np.zeros((1, n))
            coef[:, : self.clf.coef_.shape[1]] = self.clf.coef_
            self.clf.coef_ = coef

    def _vectorise(self, batch: List[Dict[str, Any]]):
        for rec in batch:
            self._ensure_features(rec.keys())
        X = [[float(rec.get(f, 0.0)) for f in self.feature_names] for rec in batch]
        y = [int(rec["y"]) for rec in batch]
        return np.asarray(X), np.asarray(y)

    def update(self, batch: List[Dict[str, Any]]) -> None:
        with tracer.start_as_current_span("train_batch") as span:
            X, y = self._vectorise(batch)
            if not hasattr(self.clf, "classes_"):
                self.clf.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                self.clf.partial_fit(X, y)
            coef = self.clf.coef_[0].tolist()
            intercept = float(self.clf.intercept_[0])
            changed = self._prev_coef != coef + [intercept]
            if changed:
                self._prev_coef = coef + [intercept]
                self._save()
                self._maybe_generate()
            ctx = span.get_span_context()
            logger.info(
                {
                    "event": "batch_update",
                    "size": len(batch),
                    "coefficients_changed": changed,
                },
                extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id},
            )

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------
    def tail_csv(self, path: Path) -> None:
        """Continuously follow ``path`` for new rows."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pos = 0
        batch: List[Dict[str, Any]] = []
        while True:
            load = psutil.cpu_percent(interval=None)
            if load > 80.0:
                time.sleep(3.0)
            if path.exists():
                with path.open() as f:
                    f.seek(pos)
                    reader = csv.DictReader(f)
                    for row in reader:
                        pos = f.tell()
                        if "y" not in row and "label" not in row:
                            continue
                        row["y"] = row.get("y") or row.get("label")
                        batch.append(row)
                        if len(batch) >= self.batch_size:
                            load = psutil.cpu_percent(interval=None)
                            if load > 80.0:
                                time.sleep(3.0)
                            self.update(batch)
                            batch.clear()
            if batch:
                load = psutil.cpu_percent(interval=None)
                if load > 80.0:
                    time.sleep(3.0)
                self.update(batch)
                batch.clear()
            time.sleep(1.0)

    async def consume_socket(self, host: str, port: int) -> None:
        reader, _ = await asyncio.open_connection(host, port)
        batch: List[Dict[str, Any]] = []
        while True:
            line = await reader.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            try:
                rec = json.loads(line.decode())
            except Exception:
                continue
            batch.append(rec)
            if len(batch) >= self.batch_size:
                load = psutil.cpu_percent(interval=None)
                if load > 80.0:
                    await asyncio.sleep(3.0)
                self.update(batch)
                batch.clear()


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Online incremental trainer")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=Path, help="Path to trades_raw.csv to follow")
    g.add_argument("--socket", help="host:port for JSON line stream")
    p.add_argument("--model", type=Path, default=Path("model.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--no-generate",
        action="store_true",
        help="Do not run generate_mql4_from_model.py on updates",
    )
    args = p.parse_args(argv)

    trainer = OnlineTrainer(args.model, args.batch_size, not args.no_generate)
    _sd_notify_ready()
    _start_watchdog_thread()
    if args.csv:
        trainer.tail_csv(args.csv)
    else:
        host, port = args.socket.split(":", 1)
        asyncio.run(trainer.consume_socket(host, int(port)))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

