import socket
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.socket_log_service import listen_once
from proto import observer_pb2, trade_event_pb2


def test_logging_roundtrip(tmp_path: Path):
    host = "127.0.0.1"
    srv = socket.socket()
    srv.bind((host, 0))
    port = srv.getsockname()[1]
    srv.close()

    out_file = tmp_path / "trade.csv"
    t = threading.Thread(target=listen_once, args=(host, port, out_file), kwargs={"binary": True})
    t.start()
    time.sleep(0.1)

    event = trade_event_pb2.TradeEvent(
        event_id=1,
        event_time="2024.01.01 00:00:00",
        action="OPEN",
        ticket=1,
        symbol="EURUSD",
        order_type=0,
        lots=0.1,
    )
    envelope = observer_pb2.ObserverMessage(schema_version="1.0", event=event)
    payload = envelope.SerializeToString()
    packet = len(payload).to_bytes(4, "little") + payload

    client = socket.socket()
    client.connect((host, port))
    client.sendall(packet)
    client.close()

    t.join(timeout=2)
    assert out_file.exists()
    text = out_file.read_text().splitlines()
    assert text[0].startswith("event_id")
    assert "EURUSD" in text[1]
