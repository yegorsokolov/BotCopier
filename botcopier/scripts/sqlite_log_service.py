#!/usr/bin/env python3
"""Service that logs JSON events to a SQLite database."""

import argparse
import asyncio
import json
import signal
import sqlite3
from contextlib import closing
from pathlib import Path
from asyncio import StreamReader, StreamWriter, Queue

FIELDS = [
    "schema_version",
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
    "profit_after_trade",
    "spread",
    "comment",
    "remaining_lots",
    "slippage",
    "volume",
    "open_time",
    "book_bid_vol",
    "book_ask_vol",
    "book_imbalance",
    "sl_hit_dist",
    "tp_hit_dist",
    "decision_id",
    "equity",
    "margin_level",
]


async def _writer_task(db_file: Path, queue: Queue, batch_size: int) -> None:
    """Write queued rows to ``db_file``."""

    db_file.parent.mkdir(parents=True, exist_ok=True)
    cols = ",".join(FIELDS)
    placeholders = ",".join(["?"] * len(FIELDS))
    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS logs ({','.join([f'{c} TEXT' for c in FIELDS])})"
        )
        # Improve write performance on slow disks
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        insert_sql = f"INSERT INTO logs ({cols}) VALUES ({placeholders})"
        pending: list[list[str]] = []
        try:
            while True:
                row = await queue.get()
                if row is None:
                    break
                pending.append([row.get(f, "") for f in FIELDS])
                if len(pending) >= batch_size:
                    conn.executemany(insert_sql, pending)
                    conn.commit()
                    pending.clear()
                queue.task_done()
        finally:
            if pending:
                conn.executemany(insert_sql, pending)
                conn.commit()


async def _handle_conn(reader: StreamReader, writer: StreamWriter, queue: Queue) -> None:
    """Process one connection and push rows to queue."""

    try:
        while data := await reader.readline():
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            await queue.put(obj)
    finally:
        writer.close()
        await writer.wait_closed()


def listen_once(host: str, port: int, db_file: Path, batch_size: int = 100) -> None:
    """Accept a single connection and process it until closed."""

    async def _run() -> None:
        queue: Queue = Queue()
        done = asyncio.Event()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        async def wrapped(reader: StreamReader, writer: StreamWriter) -> None:
            await _handle_conn(reader, writer, queue)
            done.set()

        server = await asyncio.start_server(wrapped, host, port)
        writer_task = asyncio.create_task(_writer_task(db_file, queue, batch_size))
        async with server:
            await asyncio.wait(
                [done.wait(), stop.wait()], return_when=asyncio.FIRST_COMPLETED
            )

        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def serve(host: str, port: int, db_file: Path, batch_size: int = 100) -> None:
    """Accept multiple connections concurrently."""

    async def _run() -> None:
        queue: Queue = Queue()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        server = await asyncio.start_server(
            lambda r, w: _handle_conn(r, w, queue), host, port
        )
        writer_task = asyncio.create_task(_writer_task(db_file, queue, batch_size))
        async with server:
            await stop.wait()

        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser(description="Listen on socket and insert into SQLite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--db", required=True, help="output SQLite file")
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="number of rows to buffer before committing",
    )
    args = p.parse_args()

    serve(args.host, args.port, Path(args.db), args.batch_size)


if __name__ == "__main__":
    main()
