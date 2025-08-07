import socket
import threading
import time
import json
import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.socket_log_service import listen_once


def test_stream_listener(tmp_path: Path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    out_file = tmp_path / "out.csv.gz"

    t = threading.Thread(target=listen_once, args=(host, port, out_file))
    t.start()
    time.sleep(0.1)

    decision_msg = {
        "event_id": 1,
        "event_time": "t",
        "broker_time": "b",
        "local_time": "l",
        "action": "DECISION",
        "ticket": 0,
        "magic": 0,
        "source": "mt4",
        "symbol": "EURUSD",
        "order_type": 0,
        "lots": 0.0,
        "price": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "profit": 0.0,
        "comment": "",
        "remaining_lots": 0.0,
    }
    trade_msg = {
        "event_id": 2,
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
        "decision_id": 1,
    }

    client = socket.socket()
    client.connect((host, port))
    client.sendall(json.dumps(decision_msg).encode() + b"\n")
    client.sendall(json.dumps(trade_msg).encode() + b"\n")
    client.close()

    t.join(timeout=2)
    assert not t.is_alive()

    with gzip.open(out_file, "rt") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    header = lines[0].split(";")
    assert "decision_id" in header
    assert "trace_id" in header and "span_id" in header
    rows = [dict(zip(header, l.split(";"))) for l in lines[1:]]
    assert rows[1]["symbol"] == "EURUSD"
    assert rows[1]["decision_id"] == rows[0]["event_id"]
    assert len(rows[1]["trace_id"]) == 32
    assert len(rows[1]["span_id"]) == 16
