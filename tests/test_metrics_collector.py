import socket
import threading
import time
import json
import sqlite3
from pathlib import Path
import urllib.request
import sys
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.metrics_collector import _handle_conn, _writer_task, FIELDS
from aiohttp import web


def _run_server_once(host: str, port: int, db_file: Path, http_port: int, stop_evt: threading.Event) -> None:
    async def run() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        server = await asyncio.start_server(
            lambda r, w: _handle_conn(r, w, queue), host, port
        )
        writer_task = asyncio.create_task(_writer_task(db_file, queue))

        async def metrics_handler(request: web.Request) -> web.Response:
            conn = sqlite3.connect(db_file)
            try:
                rows = conn.execute(
                    "SELECT * FROM metrics ORDER BY ROWID DESC LIMIT ?",
                    (100,),
                ).fetchall()
            finally:
                conn.close()
            rows = [dict(zip(FIELDS, r)) for r in reversed(rows)]
            return web.json_response(rows)

        app = web.Application()
        app.add_routes([web.get("/metrics", metrics_handler)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, http_port)
        await site.start()

        async with server:
            await asyncio.get_running_loop().run_in_executor(None, stop_evt.wait)

        await runner.cleanup()
        await queue.join()
        queue.put_nowait(None)
        await writer_task

    asyncio.run(run())


def test_metrics_collector(tmp_path: Path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    http_sock = socket.socket()
    http_sock.bind((host, 0))
    http_port = http_sock.getsockname()[1]
    http_sock.close()

    db_file = tmp_path / "metrics.db"
    stop_evt = threading.Event()
    t = threading.Thread(target=_run_server_once, args=(host, port, db_file, http_port, stop_evt))
    t.start()
    time.sleep(0.1)

    msg = {
        "type": "metrics",
        "time": "t",
        "magic": 1,
        "win_rate": 0.5,
        "avg_profit": 1.0,
        "trade_count": 2,
        "drawdown": 0.1,
        "sharpe": 1.5,
        "sortino": 1.2,
        "expectancy": 0.7,
        "file_write_errors": 0,
        "socket_errors": 0,
    }

    with socket.create_connection((host, port)) as client:
        client.sendall(json.dumps(msg).encode() + b"\n")

    time.sleep(0.2)
    with urllib.request.urlopen(f"http://{host}:{http_port}/metrics") as resp:
        data = json.load(resp)
    assert len(data) == 1
    assert data[0]["magic"] == "1"
    assert "sortino" in data[0]
    assert "expectancy" in data[0]

    stop_evt.set()
    t.join(timeout=2)
    assert not t.is_alive()

    conn = sqlite3.connect(db_file)
    try:
        rows = conn.execute("SELECT * FROM metrics").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1

