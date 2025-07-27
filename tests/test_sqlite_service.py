import socket
import threading
import time
import json
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sqlite_log_service import listen_once
from scripts.train_target_clone import _load_logs


def test_sqlite_log_service(tmp_path: Path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    db_file = tmp_path / "stream.db"

    t = threading.Thread(target=listen_once, args=(host, port, db_file))
    t.start()
    time.sleep(0.1)

    msg = {
        "event_id": 1,
        "event_time": "t",
        "broker_time": "b",
        "local_time": "l",
        "action": "OPEN",
        "ticket": 1,
        "magic": 2,
        "source": "mt4",
        "symbol": "EURUSD",
        "order_type": 0,
        "lots": 0.1,
        "price": 1.2345,
        "sl": 1.0,
        "tp": 2.0,
        "profit": 0.0,
        "comment": "hi",
        "remaining_lots": 0.1,
    }

    client = socket.socket()
    client.connect((host, port))
    client.sendall(json.dumps(msg).encode() + b"\n")
    client.close()

    t.join(timeout=2)
    assert not t.is_alive()

    conn = sqlite3.connect(db_file)
    try:
        rows = conn.execute("SELECT * FROM logs").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1


def test_load_logs_from_db(tmp_path: Path):
    db_file = tmp_path / "logs.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        "CREATE TABLE logs (event_id TEXT, event_time TEXT, broker_time TEXT, local_time TEXT, action TEXT, ticket TEXT, magic TEXT, source TEXT, symbol TEXT, order_type TEXT, lots TEXT, price TEXT, sl TEXT, tp TEXT, profit TEXT, comment TEXT, remaining_lots TEXT)"
    )
    conn.execute(
        "INSERT INTO logs VALUES (1, '2024.01.01 00:00:00', '', '', 'OPEN', '1', '', '', 'EURUSD', '0', '0.1', '1.1000', '1.0950', '1.1100', '0', '', '0.1')"
    )
    conn.commit()
    conn.close()

    df = _load_logs(db_file)
    assert not df.empty
    assert "symbol" in df.columns

