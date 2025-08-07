#!/usr/bin/env python3
"""Service that listens for protobuf log events and appends them to a CSV file."""

import argparse
import asyncio
import csv
import os
import gzip
from pathlib import Path
from asyncio import StreamReader, StreamWriter, Queue
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

from google.protobuf.json_format import MessageToDict

from proto import observer_pb2

FIELDS = [
    "event_id",
    "event_time",
    "broker_time",
    "local_time",
    "action",
    "ticket",
    "magic",
    "source",
    "symbol",
    "order_type",
    "lots",
    "price",
    "sl",
    "tp",
    "profit",
    "comment",
    "remaining_lots",
    "decision_id",
    "trace_id",
    "span_id",
]

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "socket_log_service")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

async def _writer_task(out_file: Path, queue: Queue) -> None:
    """Write rows from ``queue`` to ``out_file``."""

    out_file.parent.mkdir(parents=True, exist_ok=True)
    open_func = gzip.open if out_file.suffix == ".gz" else open
    with open_func(out_file, "at", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        if f.tell() == 0:
            writer.writerow(FIELDS)

        while True:
            row = await queue.get()
            if row is None:
                break
            writer.writerow(row)
            f.flush()
            queue.task_done()


async def _handle_conn(
    reader: StreamReader,
    writer: StreamWriter,
    queue: Queue,
    binary: bool = False,
) -> None:
    """Process a single connection and push rows to ``queue``."""

    try:
        if binary:
            while True:
                try:
                    header = await reader.readexactly(4)
                except asyncio.IncompleteReadError:
                    break
                length = int.from_bytes(header, "little")
                try:
                    payload = await reader.readexactly(length)
                except asyncio.IncompleteReadError:
                    break
                try:
                    msg = observer_pb2.ObserverMessage.FromString(payload)
                except Exception:
                    continue
                kind = msg.WhichOneof("payload")
                if kind != "event":
                    continue
                obj = MessageToDict(msg.event, preserving_proto_field_name=True)
                with tracer.start_as_current_span("log_event") as span:
                    ctx = span.get_span_context()
                    obj.setdefault("trace_id", format_trace_id(ctx.trace_id))
                    obj.setdefault("span_id", format_span_id(ctx.span_id))
                    row = [str(obj.get(field, "")) for field in FIELDS]
                    await queue.put(row)
        else:
            while data := await reader.readline():
                if not data:
                    break
                try:
                    msg = observer_pb2.ObserverMessage.FromString(data.strip())
                except Exception:
                    continue
                kind = msg.WhichOneof("payload")
                if kind != "event":
                    continue
                obj = MessageToDict(msg.event, preserving_proto_field_name=True)
                with tracer.start_as_current_span("log_event") as span:
                    ctx = span.get_span_context()
                    obj.setdefault("trace_id", format_trace_id(ctx.trace_id))
                    obj.setdefault("span_id", format_span_id(ctx.span_id))
                    row = [str(obj.get(field, "")) for field in FIELDS]
                    await queue.put(row)
    finally:
        writer.close()
        await writer.wait_closed()


def listen_once(host: str, port: int, out_file: Path, *, binary: bool = False) -> None:
    """Accept a single connection and process it until closed."""

    async def _run() -> None:
        queue: Queue = Queue()
        done = asyncio.Event()

        async def wrapped(reader: StreamReader, writer: StreamWriter) -> None:
            await _handle_conn(reader, writer, queue, binary=binary)
            done.set()

        server = await asyncio.start_server(wrapped, host, port)
        writer_task = asyncio.create_task(_writer_task(out_file, queue))
        async with server:
            await done.wait()

        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def serve(host: str, port: int, out_file: Path, *, binary: bool = False) -> None:
    """Accept multiple connections concurrently and append lines."""

    async def _run() -> None:
        queue: Queue = Queue()

        server = await asyncio.start_server(
            lambda r, w: _handle_conn(r, w, queue, binary=binary),
            host,
            port,
        )
        writer_task = asyncio.create_task(_writer_task(out_file, queue))
        async with server:
            await server.serve_forever()

        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser(description="Listen on socket and append to CSV")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--out", required=True, help="output CSV file")
    p.add_argument("--binary", action="store_true", help="expect length-prefixed protobuf")
    args = p.parse_args()

    serve(args.host, args.port, Path(args.out), binary=args.binary)


if __name__ == "__main__":
    main()
