from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from dashboard import server


@pytest.fixture(autouse=True)
def _reset_state():
    original_token = server.API_TOKEN
    server.API_TOKEN = ""
    server.reset_state()
    yield
    server.API_TOKEN = original_token
    server.reset_state()


def _client() -> TestClient:
    return TestClient(server.app)


def test_controls_default_state():
    with _client() as client:
        resp = client.get("/controls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["auto_trading"] is True
        assert data["auto_retraining"] is True
        assert data["shadow_mode"] is True
        assert data["emergency_stop"] is False
        assert "last_updated" in data


def test_controls_update_emergency_stop():
    with _client() as client:
        resp = client.post("/controls", json={"emergency_stop": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["emergency_stop"] is True
        assert data["auto_trading"] is False

        resp = client.post("/controls", json={"emergency_stop": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["emergency_stop"] is False
        assert data["auto_trading"] is True


def test_shadow_status_without_history():
    with _client() as client:
        resp = client.get("/shadow_status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_month"] is False
        assert data["meets_threshold"] is False
        assert data["current_accuracy"] is None


def test_shadow_status_meets_threshold():
    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=31)
    server._record_shadow_accuracy(  # type: ignore[attr-defined]
        {"metrics": {"shadow_prediction_accuracy": 0.96, "time": month_ago.isoformat()}}
    )
    server._record_shadow_accuracy(  # type: ignore[attr-defined]
        {
            "metrics": {
                "shadow_prediction_accuracy": 0.96,
                "time": (now - timedelta(days=15)).isoformat(),
            }
        }
    )
    server._record_shadow_accuracy(  # type: ignore[attr-defined]
        {"metrics": {"shadow_prediction_accuracy": 0.97, "time": now.isoformat()}}
    )

    with _client() as client:
        resp = client.get("/shadow_status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_month"] is True
        assert data["meets_threshold"] is True
        assert data["current_accuracy"] == pytest.approx(0.97, rel=1e-6)
        assert data["window_min_accuracy"] == pytest.approx(0.96, rel=1e-6)
