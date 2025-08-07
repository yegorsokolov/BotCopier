#!/usr/bin/env python3
"""Listen for protobuf messages from the observer EA and append them to CSV logs.

Messages are expected to be length-prefixed ``ObserverMessage`` protobufs. Each
envelope carries a ``schema_version`` and either a ``TradeEvent`` or ``Metrics``
payload. Records are written directly to the appropriate CSV file under
``logs/``. Set the ``SCHEMA_VERSION`` environment variable to override the
default version.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import platform
import pkgutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

from google.protobuf.json_format import MessageToDict

from proto import observer_pb2

# Expected schema version for incoming messages. Can be overridden via env var.
EXPECTED_SCHEMA_VERSION = os.environ.get("SCHEMA_VERSION", "1.0")

# Mapping from message type to output CSV file.
LOG_FILES = {
    "event": Path("logs/trades_raw.csv"),
    "metric": Path("logs/metrics.csv"),
}

RUN_INFO_PATH = Path("logs/run_info.json")
run_info_written = False


resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "stream_listener")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


def append_csv(path: Path, record: dict) -> None:
    """Append ``record`` to ``path`` writing a header row when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def process_message(envelope: observer_pb2.ObserverMessage) -> None:
    """Validate and route a message to the appropriate CSV file."""
    if envelope.schema_version != EXPECTED_SCHEMA_VERSION:
        print(
            f"Unsupported schema_version: {envelope.schema_version}; expected {EXPECTED_SCHEMA_VERSION}",
            file=sys.stderr,
        )
        return

    kind = envelope.WhichOneof("payload")
    if kind not in LOG_FILES:
        print(f"Unknown message type: {kind}", file=sys.stderr)
        return
    global run_info_written
    if not run_info_written:
        info = {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "libraries": sorted(m.name for m in pkgutil.iter_modules()),
        }
        RUN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_INFO_PATH.open("w") as f:
            json.dump(info, f, indent=2)
        run_info_written = True
    with tracer.start_as_current_span(f"process_{kind}") as span:
        if kind == "event":
            record = MessageToDict(envelope.event, preserving_proto_field_name=True)
        else:
            record = MessageToDict(envelope.metric, preserving_proto_field_name=True)
        ctx = span.get_span_context()
        record.setdefault("trace_id", format_trace_id(ctx.trace_id))
        record["span_id"] = format_span_id(ctx.span_id)
        append_csv(LOG_FILES[kind], record)


def main() -> int:
    p = argparse.ArgumentParser(description="Process observer stream logs")
    p.add_argument(
        "--binary",
        action="store_true",
        help="expect length-prefixed protobuf records",
    )
    args = p.parse_args()

    if args.binary:
        buf = sys.stdin.buffer
        while True:
            header = buf.read(4)
            if len(header) < 4:
                break
            length = int.from_bytes(header, "little")
            payload = buf.read(length)
            if len(payload) < length:
                break
            try:
                envelope = observer_pb2.ObserverMessage.FromString(payload)
            except Exception as exc:
                print(f"Invalid protobuf: {exc}", file=sys.stderr)
                continue
            process_message(envelope)
    else:
        # Treat stdin as a stream of delimited protobuf messages.
        data = sys.stdin.buffer.read()
        offset = 0
        while offset < len(data):
            if offset + 4 > len(data):
                break
            length = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            payload = data[offset : offset + length]
            offset += length
            try:
                envelope = observer_pb2.ObserverMessage.FromString(payload)
            except Exception as exc:
                print(f"Invalid protobuf: {exc}", file=sys.stderr)
                continue
            process_message(envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
