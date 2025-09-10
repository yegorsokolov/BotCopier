#!/usr/bin/env python3
"""Package generated artefacts and push them to GitHub.

After each trading or training session the raw outputs ``trades_raw.csv``,
``metrics.csv`` and ``model.json`` are compressed into a single
``run_<timestamp>.tar.gz`` archive.  A ``manifest.json`` containing the current
commit hash, schema version and SHA256 checksums of each file is included in the
archive.  The archive is committed to the repository and the raw files are
removed locally. Authentication is performed using the ``GITHUB_TOKEN``
environment variable which must contain a personal access token with permission
to push to the repository.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_trace_id, format_span_id

REPO_ROOT = Path(__file__).resolve().parents[1]

if os.name != "nt" and (xdg := os.getenv("XDG_DATA_HOME")):
    LOG_DIR = Path(xdg) / "botcopier" / "logs"
else:
    LOG_DIR = REPO_ROOT / "logs"

TRADES_FILE = LOG_DIR / "trades_raw.csv"
METRICS_FILE = LOG_DIR / "metrics.csv"
MODEL_FILE = REPO_ROOT / "model.json"


resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "upload_logs")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint)))
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log: dict[str, Any] = {"level": record.levelname}
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


def _log(msg: dict[str, Any]) -> None:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    extra = {"trace_id": ctx.trace_id, "span_id": ctx.span_id}
    logger.info(msg, extra=extra)


def run(cmd: list[str]) -> None:
    with tracer.start_as_current_span("run_cmd") as span:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        _log({"cmd": cmd})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_archive() -> Path | None:
    """Create a ``run_<timestamp>.tar.gz`` archive with a manifest."""

    files = [TRADES_FILE, METRICS_FILE, MODEL_FILE]
    if not all(p.exists() for p in files):
        return None

    with tracer.start_as_current_span("create_archive") as span:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
        schema_version = os.environ.get("SCHEMA_VERSION", "1.0")
        manifest = {
            "commit": commit,
            "schema_version": schema_version,
            "files": {p.name: {"sha256": _sha256(p)} for p in files},
        }
        manifest_path = LOG_DIR / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archive = LOG_DIR / f"run_{timestamp}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for p in files + [manifest_path]:
                tar.add(p, arcname=p.name)

        for p in files + [manifest_path]:
            p.unlink()

        _log({"event": "archive_created", "path": str(archive)})
        return archive


def main() -> int:
    with tracer.start_as_current_span("upload_logs"):
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            print("GITHUB_TOKEN environment variable is required", file=sys.stderr)
            _log({"error": "missing_token"})
            return 1

        archive = create_archive()
        if not archive:
            print("Required log files are missing", file=sys.stderr)
            _log({"warning": "missing_logs"})
            return 0

        run(["git", "add", str(archive)])
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain", str(archive)], cwd=REPO_ROOT
            )
            .decode()
            .strip()
        )
        if not status:
            print("No changes to commit")
            _log({"info": "no_changes"})
            return 0

        commit_message = f"upload logs {dt.date.today().isoformat()}"
        run(["git", "commit", "-m", commit_message])

        origin_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT
        ).decode().strip()
        if origin_url.startswith("https://"):
            push_url = origin_url.replace("https://", f"https://{token}@")
        elif origin_url.startswith("git@github.com:"):
            repo_path = origin_url.split(":", 1)[1]
            push_url = f"https://{token}@github.com/{repo_path}"
        else:
            push_url = origin_url

        run(["git", "push", push_url, "HEAD"])
        _log({"event": "logs_uploaded", "archive": str(archive)})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
