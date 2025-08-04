import socket
import threading
import time
import json
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

    out_file = tmp_path / "out.csv"

    t = threading.Thread(target=listen_once, args=(host, port, out_file))
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

    with open(out_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    header = lines[0].split(";")
    assert "trace_id" in header and "span_id" in header
    values = dict(zip(header, lines[1].split(";")))
    assert values["symbol"] == "EURUSD"
    assert len(values["trace_id"]) == 32
    assert len(values["span_id"]) == 16
