#!/usr/bin/env python3
"""Service that listens for JSON log events and appends them to a CSV file."""

import argparse
import asyncio
import csv
import json
from pathlib import Path
from asyncio import StreamReader, StreamWriter, Queue

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
]


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


async def _handle_conn(reader: StreamReader, writer: StreamWriter, queue: Queue) -> None:
    """Process a single connection and push rows to ``queue``."""

    try:
        while data := await reader.readline():
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = [str(obj.get(field, "")) for field in FIELDS]
            await queue.put(row)
    finally:
        writer.close()
        await writer.wait_closed()


def listen_once(host: str, port: int, out_file: Path) -> None:
    """Accept a single connection and process it until closed."""

    async def _run() -> None:
        queue: Queue = Queue()
        done = asyncio.Event()

        async def wrapped(reader: StreamReader, writer: StreamWriter) -> None:
            await _handle_conn(reader, writer, queue)
            done.set()

        server = await asyncio.start_server(wrapped, host, port)
        writer_task = asyncio.create_task(_writer_task(out_file, queue))
        async with server:
            await done.wait()

        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def serve(host: str, port: int, out_file: Path) -> None:
    """Accept multiple connections concurrently and append lines."""

    async def _run() -> None:
        queue: Queue = Queue()

        server = await asyncio.start_server(
            lambda r, w: _handle_conn(r, w, queue), host, port
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
    args = p.parse_args()

    serve(args.host, args.port, Path(args.out))


if __name__ == "__main__":
    main()
