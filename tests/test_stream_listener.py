import socket
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stream_listener import listen_once


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

    client = socket.socket()
    client.connect((host, port))
    client.sendall(b"a;b;c\n")
    client.close()

    t.join(timeout=2)
    assert not t.is_alive()

    with open(out_file) as f:
        content = f.read().strip()
    assert content == "a;b;c"
