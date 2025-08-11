import socket
import threading
import numpy as np

from scripts.federated_buffer import FederatedBufferClient, serve


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_two_clients_aggregate() -> None:
    port = _free_port()
    server = serve(f"127.0.0.1:{port}")
    t = threading.Thread(target=server.wait_for_termination, daemon=True)
    t.start()

    c1 = FederatedBufferClient(f"127.0.0.1:{port}")
    c2 = FederatedBufferClient(f"127.0.0.1:{port}")
    e1 = [(np.array([1.0], dtype=np.float32), 0, 1.0, np.array([0.0], dtype=np.float32))]
    e2 = [(np.array([3.0], dtype=np.float32), 1, -1.0, np.array([2.0], dtype=np.float32))]
    c1.upload(e1)
    c2.upload(e2)

    agg = c1.download()
    assert len(agg) == 2
    rewards = [exp[2] for exp in agg]
    assert abs(float(np.mean(rewards))) < 0.5

    server.stop(0)
    t.join(timeout=5)
