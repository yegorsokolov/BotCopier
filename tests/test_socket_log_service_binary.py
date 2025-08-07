import socket
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.socket_log_service import listen_once
from proto import observer_pb2, trade_event_pb2


def test_socket_log_service_binary(tmp_path: Path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    out_file = tmp_path / "out.csv"
    t = threading.Thread(target=listen_once, args=(host, port, out_file), kwargs={"binary": True})
    t.start()
    time.sleep(0.1)

    event = trade_event_pb2.TradeEvent(
        event_id=1,
        event_time="t",
        broker_time="b",
        local_time="l",
        action="OPEN",
        ticket=1,
        magic=2,
        source="mt4",
        symbol="EURUSD",
        order_type=0,
        lots=0.1,
        price=1.2345,
        sl=1.0,
        tp=2.0,
        profit=0.0,
        comment="hi",
        remaining_lots=0.1,
    )
    env = observer_pb2.ObserverMessage(schema_version="1.0", event=event)
    payload = env.SerializeToString()
    packet = len(payload).to_bytes(4, "little") + payload

    client = socket.socket()
    client.connect((host, port))
    client.sendall(packet)
    client.close()

    t.join(timeout=2)
    assert out_file.exists()
    lines = [l.strip() for l in out_file.read_text().splitlines() if l.strip()]
    assert "EURUSD" in lines[-1]
