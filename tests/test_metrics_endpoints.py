import socket
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("prometheus_client")

try:  # optional dependency in minimal environments
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - fastapi not installed
    TestClient = None  # type: ignore[assignment]

from botcopier.metrics import ERROR_COUNTER, TRADE_COUNTER, start_metrics_server
from botcopier.scripts import bandit_router


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fetch_metrics(port: int, timeout: float = 5.0) -> str:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/metrics"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url) as response:  # nosec B310 - local fetch
                return response.read().decode("utf-8")
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.05)
    raise AssertionError("metrics endpoint did not become ready")


def test_metrics_server_exposes_counters():
    port = _find_free_port()
    start_metrics_server(port)
    TRADE_COUNTER.inc(3)
    ERROR_COUNTER.labels(type="test").inc()
    payload = _fetch_metrics(port)
    assert "botcopier_trades_total" in payload
    assert "botcopier_errors_total" in payload


@pytest.mark.skipif(TestClient is None, reason="fastapi is not installed")
def test_bandit_router_metrics_endpoint(tmp_path):
    state_file = tmp_path / "state.json"
    router = bandit_router.BanditRouter(models=1, method="thompson", state_file=str(state_file))
    app = bandit_router.create_app(router)
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "botcopier_latency_seconds_bucket" in response.text
