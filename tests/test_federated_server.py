import json
import socket
import threading
import time
from pathlib import Path

import requests
import uvicorn

from scripts import federated_server
from botcopier.training.pipeline import sync_with_server


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        federated_server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    return server, thread


def test_two_clients_sync(tmp_path: Path) -> None:
    port = _free_port()
    server, thread = _start_server(port)
    federated_server.state["expected"] = 2
    url = f"http://127.0.0.1:{port}"

    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps({"coefficients": [1.0, 2.0], "intercept": 0.5}))
    m2.write_text(json.dumps({"coefficients": [3.0, 4.0], "intercept": 1.5}))

    t1 = threading.Thread(
        target=sync_with_server, args=(m1, url), kwargs={"poll_interval": 0.1}
    )
    t2 = threading.Thread(
        target=sync_with_server, args=(m2, url), kwargs={"poll_interval": 0.1}
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    r = requests.get(f"{url}/weights")
    data = r.json()
    assert data["weights"] == [2.0, 3.0]
    assert data["intercept"] == 1.0

    with open(m1) as f:
        m1_data = json.load(f)
    with open(m2) as f:
        m2_data = json.load(f)
    assert m1_data["coefficients"] == m2_data["coefficients"] == [2.0, 3.0]
    assert m1_data["intercept"] == m2_data["intercept"] == 1.0

    server.should_exit = True
    thread.join(timeout=5)

