import json
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def test_sqlite_log_service_handles_sigterm(tmp_path):
    db = tmp_path / "logs.db"
    script = (
        Path(__file__).resolve().parents[1]
        / "botcopier"
        / "scripts"
        / "sqlite_log_service.py"
    )
    port = 8765
    proc = subprocess.Popen(
        [sys.executable, str(script), "--db", str(db), "--port", str(port)],
    )
    try:
        time.sleep(1.0)
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(json.dumps({"event_id": "1"}).encode() + b"\n")
        sock.close()
        time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    finally:
        proc.kill()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    assert rows == 1
