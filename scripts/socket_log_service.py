#!/usr/bin/env python3
"""Service that listens for JSON log events and appends them to a CSV file."""

import argparse
import asyncio
import csv
import json
import os
import zlib
from pathlib import Path
from asyncio import StreamReader, StreamWriter, Queue

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

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
    with open(out_file, "a", newline="") as f:
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
                    data = zlib.decompress(payload).decode("utf-8")
                    obj = json.loads(data)
                except Exception:
                    continue
                with tracer.start_as_current_span("log_event") as span:
                    ctx = span.get_span_context()
                    obj.setdefault("trace_id", format_trace_id(ctx.trace_id))
                    obj.setdefault("span_id", format_span_id(ctx.span_id))
                    row = [str(obj.get(field, "")) for field in FIELDS]
                    await queue.put(row)
        else:
            while data := await reader.readline():
                with tracer.start_as_current_span("log_event") as span:
                    line = data.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
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
    p.add_argument("--binary", action="store_true", help="expect length-prefixed gzipped JSON")
    args = p.parse_args()

    serve(args.host, args.port, Path(args.out), binary=args.binary)


if __name__ == "__main__":
    main()
