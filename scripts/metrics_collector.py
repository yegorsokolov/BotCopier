#!/usr/bin/env python3
"""Listen for metric messages and store them in a SQLite database."""

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
from asyncio import StreamReader, StreamWriter, Queue

FIELDS = [
    "time",
    "magic",
    "win_rate",
    "avg_profit",
    "trade_count",
    "drawdown",
    "sharpe",
    "file_write_errors",
    "socket_errors",
]


async def _writer_task(db_file: Path, queue: Queue) -> None:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    cols = ",".join(FIELDS)
    placeholders = ",".join(["?"] * len(FIELDS))
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS metrics ({','.join([f'{c} TEXT' for c in FIELDS])})"
    )
    insert_sql = f"INSERT INTO metrics ({cols}) VALUES ({placeholders})"
    try:
        while True:
            row = await queue.get()
            if row is None:
                break
            conn.execute(insert_sql, [row.get(f, "") for f in FIELDS])
            conn.commit()
            queue.task_done()
    finally:
        conn.close()


async def _handle_conn(reader: StreamReader, writer: StreamWriter, queue: Queue) -> None:
    try:
        while data := await reader.readline():
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "metrics":
                await queue.put(obj)
    finally:
        writer.close()
        await writer.wait_closed()


def serve(host: str, port: int, db_file: Path) -> None:
    async def _run() -> None:
        queue: Queue = Queue()
        server = await asyncio.start_server(lambda r, w: _handle_conn(r, w, queue), host, port)
        writer_task = asyncio.create_task(_writer_task(db_file, queue))
        async with server:
            await server.serve_forever()
        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser(description="Collect metric messages into SQLite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--db", required=True, help="output SQLite file")
    args = p.parse_args()

    serve(args.host, args.port, Path(args.db))


if __name__ == "__main__":
    main()
