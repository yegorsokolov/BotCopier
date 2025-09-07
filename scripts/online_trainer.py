#!/usr/bin/env python3
"""Incrementally update a model from streaming trade events.

The trainer is designed to run continuously.  It tails
``logs/trades_raw.csv`` or subscribes to an Arrow Flight stream.  After each batch it updates
an :class:`~sklearn.linear_model.SGDClassifier` using :meth:`partial_fit` and
persists the coefficients to ``model.json`` for downstream consumers.

Hardware capabilities are sampled via :func:`detect_resources` to determine an
appropriate throttling level.  On lightweight VPS hosts the trainer yields
the CPU more aggressively, preventing it from overwhelming the terminal.
During processing the current load is checked with
``psutil.cpu_percent`` and the worker sleeps when the threshold is exceeded.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import os
import threading
import subprocess
import sys
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

try:  # detect resources to adapt behaviour on weaker hardware
    if __package__:
        from .train_target_clone import detect_resources  # type: ignore
    else:  # pragma: no cover - script executed directly
        from train_target_clone import detect_resources  # type: ignore
except Exception:  # pragma: no cover - detection optional
    detect_resources = None  # type: ignore

try:  # drift metrics utilities
    if __package__:
        from .drift_monitor import _compute_metrics, _update_model
    else:  # pragma: no cover - script executed directly
        from drift_monitor import _compute_metrics, _update_model  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _compute_metrics = _update_model = None  # type: ignore

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
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.clf = SGDClassifier(loss="log_loss")
        self.feature_names: List[str] = []
        self.feature_flags: Dict[str, bool] = {}
        self.model_type: str = "logreg"
        self._prev_coef: List[float] | None = None
        self.training_mode = "lite"
        self.cpu_threshold = 80.0
        self.sleep_seconds = 3.0
        self.half_life_days = 0.0
        self.weight_decay: Dict[str, Any] | None = None
        if self.model_path.exists():
            self._load()
        elif detect_resources:
            try:
                res = detect_resources()
                self.training_mode = res.get("mode", self.training_mode)
                self.feature_flags["order_book"] = res.get("heavy_mode", False)
            except Exception:
                pass
        self.feature_flags.setdefault(
            "order_book", self.training_mode not in ("lite",)
        )
        self._apply_mode()

    def _apply_mode(self) -> None:
        if self.training_mode == "lite":
            self.cpu_threshold = 50.0
            self.sleep_seconds = 6.0
        else:
            self.cpu_threshold = 80.0
            self.sleep_seconds = 3.0

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
        self.feature_flags = data.get("feature_flags", {})
        if "order_book" not in self.feature_flags:
            self.feature_flags["order_book"] = self.training_mode not in ("lite",)
        self.model_type = data.get("model_type", self.model_type)
        coef = data.get("coefficients")
        intercept = data.get("intercept")
        self.half_life_days = float(data.get("half_life_days", 0.0))
        self.weight_decay = data.get("weight_decay")
        if self.feature_names and coef is not None and intercept is not None:
            n = len(self.feature_names)
            self.clf.partial_fit(np.zeros((1, n)), [0], classes=np.array([0, 1]))
            self.clf.coef_ = np.array([coef])
            self.clf.intercept_ = np.array([intercept])
            self._prev_coef = list(coef) + [intercept]
        self._apply_mode()

    def _save(self) -> None:
        payload = {
            "feature_names": self.feature_names,
            "coefficients": self.clf.coef_[0].tolist(),
            "intercept": float(self.clf.intercept_[0]),
            "training_mode": self.training_mode,
            "mode": self.training_mode,
            "feature_flags": self.feature_flags,
            "model_type": self.model_type,
        }
        if self.half_life_days:
            payload["half_life_days"] = self.half_life_days
        if self.weight_decay:
            payload["weight_decay"] = self.weight_decay
        try:
            existing = json.loads(self.model_path.read_text())
        except Exception:
            existing = {}
        existing.update(payload)
        self.model_path.write_text(json.dumps(existing))
    # ------------------------------------------------------------------
    # Incremental training
    # ------------------------------------------------------------------
    def _ensure_features(self, keys: Iterable[str]) -> None:
        if not self.feature_flags.get("order_book", False):
            keys = [k for k in keys if not k.startswith("book_")]
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

    def _log_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            preds = self.clf.predict(X)
            acc = float(np.mean(preds == y))
        except Exception:
            acc = 0.0
        logger.info({"event": "validation", "size": len(y), "accuracy": acc})

    def update(self, batch: List[Dict[str, Any]]) -> bool:
        with tracer.start_as_current_span("train_batch") as span:
            X, y = self._vectorise(batch)
            if not hasattr(self.clf, "classes_"):
                self.clf.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                self.clf.partial_fit(X, y)
            coef = self.clf.coef_[0].tolist()
            intercept = float(self.clf.intercept_[0])
            prev = self._prev_coef
            self._prev_coef = coef + [intercept]
            changed = prev != self._prev_coef
            self._save()
            self._log_validation(X, y)
            ctx = span.get_span_context()
            logger.info(
                {
                    "event": "batch_update",
                    "size": len(batch),
                    "coefficients_changed": changed,
                },
                extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id},
            )
            return changed

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------
    def start_drift_monitor(
        self,
        baseline_file: Path,
        recent_file: Path,
        *,
        log_dir: Path,
        out_dir: Path,
        files_dir: Path,
        threshold: float = 0.2,
        interval: float = 300.0,
    ) -> None:
        """Periodically compute drift metrics and retrain if necessary."""
        if _compute_metrics is None or _update_model is None:
            logger.warning("drift_monitor unavailable")
            return

        def _loop() -> None:
            while True:
                try:
                    metrics = _compute_metrics(baseline_file, recent_file)
                    retrain = max(metrics.values()) > threshold
                    _update_model(self.model_path, metrics, retrain)
                    if retrain:
                        method = max(metrics, key=metrics.get)
                        base = Path(__file__).resolve().parent
                        subprocess.run(
                            [
                                sys.executable,
                                str(base / "auto_retrain.py"),
                                "--log-dir",
                                str(log_dir),
                                "--out-dir",
                                str(out_dir),
                                "--files-dir",
                                str(files_dir),
                                "--baseline-file",
                                str(baseline_file),
                                "--recent-file",
                                str(recent_file),
                                "--drift-method",
                                method,
                                "--drift-threshold",
                                str(threshold),
                            ],
                            check=True,
                        )
                except Exception:
                    logger.exception("drift monitoring failed")
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()

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
            if load > self.cpu_threshold:
                time.sleep(self.sleep_seconds)
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
                            if load > self.cpu_threshold:
                                time.sleep(self.sleep_seconds)
                            self.update(batch)
                            batch.clear()
            if batch:
                load = psutil.cpu_percent(interval=None)
                if load > self.cpu_threshold:
                    time.sleep(self.sleep_seconds)
                self.update(batch)
                batch.clear()
            time.sleep(1.0)

    def consume_flight(self, host: str, port: int, path: str = "trades") -> None:
        """Subscribe to an Arrow Flight stream of trade events."""
        try:  # pragma: no cover - optional dependency
            import pyarrow.flight as flight
        except Exception as exc:  # pragma: no cover - pyarrow missing
            raise RuntimeError("pyarrow is required for Flight consumption") from exc

        client = flight.FlightClient(f"grpc://{host}:{port}")
        ticket = flight.Ticket(path.encode())
        offset = 0
        batch: List[Dict[str, Any]] = []
        while True:
            load = psutil.cpu_percent(interval=None)
            if load > self.cpu_threshold:
                time.sleep(self.sleep_seconds)
                continue
            try:
                reader = client.do_get(ticket)
                table = reader.read_all()
                reader.close()
            except Exception:
                time.sleep(1.0)
                continue
            if table.num_rows > offset:
                for row in table.slice(offset).to_pylist():
                    if "y" not in row and "label" not in row:
                        continue
                    row["y"] = row.get("y") or row.get("label")
                    batch.append(row)
                    if len(batch) >= self.batch_size:
                        load = psutil.cpu_percent(interval=None)
                        if load > self.cpu_threshold:
                            time.sleep(self.sleep_seconds)
                        self.update(batch)
                        batch.clear()
                offset = table.num_rows
            if batch:
                load = psutil.cpu_percent(interval=None)
                if load > self.cpu_threshold:
                    time.sleep(self.sleep_seconds)
                self.update(batch)
                batch.clear()
            time.sleep(1.0)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Online incremental trainer")
    p.add_argument("--csv", type=Path, help="Path to trades_raw.csv to follow")
    p.add_argument("--flight-host", default="127.0.0.1", help="Arrow Flight host")
    p.add_argument("--flight-port", type=int, default=8815, help="Arrow Flight port")
    p.add_argument("--model", type=Path, default=Path("model.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--flight-path", default="trades", help="Flight path name")
    p.add_argument("--baseline-file", type=Path, help="Baseline CSV for drift monitoring")
    p.add_argument("--recent-file", type=Path, help="Recent CSV for drift monitoring")
    p.add_argument("--log-dir", type=Path, help="Log directory for retrain")
    p.add_argument("--out-dir", type=Path, help="Output directory for retrain")
    p.add_argument("--files-dir", type=Path, help="Files directory for retrain")
    p.add_argument(
        "--drift-threshold",
        type=float,
        default=float(os.getenv("DRIFT_THRESHOLD", "0.2")),
        help="Drift threshold triggering retrain",
    )
    p.add_argument(
        "--drift-interval",
        type=float,
        default=float(os.getenv("DRIFT_INTERVAL", "300")),
        help="Seconds between drift checks",
    )
    args = p.parse_args(argv)

    trainer = OnlineTrainer(args.model, args.batch_size)
    _sd_notify_ready()
    _start_watchdog_thread()
    if (
        args.baseline_file
        and args.recent_file
        and args.log_dir
        and args.out_dir
        and args.files_dir
    ):
        trainer.start_drift_monitor(
            args.baseline_file,
            args.recent_file,
            log_dir=args.log_dir,
            out_dir=args.out_dir,
            files_dir=args.files_dir,
            threshold=args.drift_threshold,
            interval=args.drift_interval,
        )
    if args.csv:
        trainer.tail_csv(args.csv)
    else:
        trainer.consume_flight(args.flight_host, args.flight_port, args.flight_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

