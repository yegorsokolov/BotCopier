from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import pyarrow.flight as flight
from pydantic import BaseModel, Field

from metrics.aggregator import add_metric, get_aggregated_metrics, reset_metrics

API_TOKEN = os.environ.get("DASHBOARD_API_TOKEN", "")
FLIGHT_URI = os.environ.get("FLIGHT_URI", "")

app = FastAPI(title="BotCopier Dashboard")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# In-memory stores
trades: List[dict] = []
metrics_store: List[dict] = []
decisions: List[dict] = []
training_progress: List[dict] = []

# Control state and automation defaults
CONTROL_FIELDS = ("auto_trading", "auto_retraining", "shadow_mode", "emergency_stop")
DEFAULT_CONTROL_STATE: Dict[str, Any] = {
    "auto_trading": True,
    "auto_retraining": True,
    "shadow_mode": True,
    "emergency_stop": False,
}
control_state: Dict[str, Any] = {**DEFAULT_CONTROL_STATE}
control_state_last_updated: datetime = datetime.now(timezone.utc)
_previous_auto_trading: Optional[bool] = None
control_lock = asyncio.Lock()

# Shadow testing accuracy tracking
SHADOW_WINDOW_DAYS = 30
SHADOW_THRESHOLD = 0.95
shadow_accuracy_history: List[dict[str, Any]] = []

# Active WebSocket connections
trade_connections: Set[WebSocket] = set()
metric_connections: Set[WebSocket] = set()
decision_connections: Set[WebSocket] = set()
training_connections: Set[WebSocket] = set()
control_connections: Set[WebSocket] = set()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _extract_shadow_accuracy(source: Dict[str, Any] | None) -> Optional[float]:
    if not isinstance(source, dict):
        return None
    for key in (
        "shadow_prediction_accuracy",
        "shadow_accuracy",
        "accuracy_shadow",
    ):
        value = source.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _record_shadow_accuracy(payload: Dict[str, Any]) -> None:
    candidates: List[Dict[str, Any]] = []
    metrics_obj = payload.get("metrics") if isinstance(payload, dict) else None
    if isinstance(metrics_obj, dict):
        candidates.append(metrics_obj)
    if isinstance(payload, dict):
        candidates.append(payload)

    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

    for candidate in candidates:
        accuracy = _extract_shadow_accuracy(candidate)
        if accuracy is None:
            continue
        timestamp = _parse_timestamp(candidate.get("time")) or _parse_timestamp(
            candidate.get("timestamp")
        )
        if timestamp is not None:
            break

    if accuracy is None:
        return

    if timestamp is None and isinstance(payload, dict):
        timestamp = _parse_timestamp(payload.get("time")) or _parse_timestamp(
            payload.get("timestamp")
        )

    if timestamp is None:
        timestamp = _now()

    shadow_accuracy_history.append({"timestamp": timestamp, "accuracy": accuracy})
    shadow_accuracy_history.sort(key=lambda entry: entry["timestamp"])

    cutoff = _now() - timedelta(days=SHADOW_WINDOW_DAYS * 2)
    while shadow_accuracy_history and shadow_accuracy_history[0]["timestamp"] < cutoff:
        shadow_accuracy_history.pop(0)


def _process_metric_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    metrics_obj = payload.get("metrics")
    if isinstance(metrics_obj, dict):
        add_metric(payload.get("strategy"), metrics_obj)
    else:
        add_metric(payload.get("strategy"), payload)
    _record_shadow_accuracy(payload)

if FLIGHT_URI:
    try:
        client = flight.FlightClient(FLIGHT_URI)
        for name, store in (("trades", trades), ("metrics", metrics_store), ("decisions", decisions)):
            desc = flight.FlightDescriptor.for_path(name)
            info = client.get_flight_info(desc)
            reader = client.do_get(info.endpoints[0].ticket)
            table = reader.read_all()
            items = table.to_pylist()
            store.extend(items)
            if name == "metrics":
                for item in items:
                    if isinstance(item, dict):
                        _process_metric_payload(item)
    except Exception:
        pass


def _control_payload() -> Dict[str, Any]:
    payload = {key: bool(control_state.get(key, DEFAULT_CONTROL_STATE[key])) for key in CONTROL_FIELDS}
    payload["last_updated"] = control_state_last_updated.isoformat()
    return payload


def _apply_control_update(update: Dict[str, Any]) -> Dict[str, Any]:
    global control_state_last_updated, _previous_auto_trading

    changed = False
    for key in CONTROL_FIELDS:
        if key not in update:
            continue
        value = update[key]
        if value is None:
            continue
        if key == "emergency_stop":
            new_value = bool(value)
            if new_value and not control_state.get("emergency_stop", False):
                _previous_auto_trading = control_state.get("auto_trading", True)
                if control_state.get("auto_trading", True):
                    control_state["auto_trading"] = False
                    changed = True
            elif not new_value and control_state.get("emergency_stop", False):
                restored = _previous_auto_trading if _previous_auto_trading is not None else True
                if control_state.get("auto_trading") != restored:
                    control_state["auto_trading"] = restored
                    changed = True
                _previous_auto_trading = None
            if control_state.get("emergency_stop") != new_value:
                control_state["emergency_stop"] = new_value
                changed = True
            continue

        bool_value = bool(value)
        if control_state.get(key) != bool_value:
            control_state[key] = bool_value
            changed = True

    if changed:
        control_state_last_updated = _now()

    return _control_payload()


async def _broadcast_controls(state: Dict[str, Any]) -> None:
    message = json.dumps(state)
    for conn in list(control_connections):
        try:
            await conn.send_text(message)
        except Exception:
            control_connections.discard(conn)


def get_shadow_status() -> Dict[str, Any]:
    if not shadow_accuracy_history:
        return {
            "has_month": False,
            "meets_threshold": False,
            "current_accuracy": None,
            "window_min_accuracy": None,
            "window_avg_accuracy": None,
            "window_days": SHADOW_WINDOW_DAYS,
            "history_length": 0,
            "status": "Shadow testing has not produced any accuracy measurements yet.",
            "threshold": SHADOW_THRESHOLD,
        }

    history_sorted = sorted(shadow_accuracy_history, key=lambda entry: entry["timestamp"])
    earliest = history_sorted[0]["timestamp"]
    latest = history_sorted[-1]["timestamp"]
    duration = latest - earliest
    has_month = duration >= timedelta(days=SHADOW_WINDOW_DAYS)

    window_start = latest - timedelta(days=SHADOW_WINDOW_DAYS)
    window_entries = [entry for entry in history_sorted if entry["timestamp"] >= window_start]
    window_min_accuracy = (
        min(entry["accuracy"] for entry in window_entries) if window_entries else None
    )
    window_avg_accuracy = (
        sum(entry["accuracy"] for entry in window_entries) / len(window_entries)
        if window_entries
        else None
    )
    meets_threshold = (
        has_month
        and window_min_accuracy is not None
        and window_min_accuracy >= SHADOW_THRESHOLD
    )

    if has_month and meets_threshold:
        status = (
            "Shadow testing indicates the MT4 code has maintained at least "
            f"{int(SHADOW_THRESHOLD * 100)}% accuracy for the last {SHADOW_WINDOW_DAYS} days."
        )
    elif not has_month:
        status = (
            "Shadow testing is still accumulating data. A full "
            f"{SHADOW_WINDOW_DAYS}-day window is required before declaring readiness."
        )
    else:
        status = (
            "Shadow testing accuracy has dipped below "
            f"{int(SHADOW_THRESHOLD * 100)}% during the most recent {SHADOW_WINDOW_DAYS}-day window."
        )

    result: Dict[str, Any] = {
        "has_month": has_month,
        "meets_threshold": bool(meets_threshold),
        "current_accuracy": history_sorted[-1]["accuracy"],
        "window_min_accuracy": window_min_accuracy,
        "window_avg_accuracy": window_avg_accuracy,
        "window_days": SHADOW_WINDOW_DAYS,
        "history_length": len(history_sorted),
        "latest_timestamp": history_sorted[-1]["timestamp"].isoformat(),
        "threshold": SHADOW_THRESHOLD,
        "status": status,
    }
    if has_month:
        result["earliest_timestamp"] = earliest.isoformat()
    return result


def reset_state() -> None:
    """Reset in-memory stores and automation state (primarily for tests)."""

    trades.clear()
    metrics_store.clear()
    decisions.clear()
    training_progress.clear()
    shadow_accuracy_history.clear()
    trade_connections.clear()
    metric_connections.clear()
    decision_connections.clear()
    training_connections.clear()
    control_connections.clear()
    control_state.clear()
    control_state.update(DEFAULT_CONTROL_STATE)
    global control_state_last_updated, _previous_auto_trading
    control_state_last_updated = _now()
    _previous_auto_trading = None
    reset_metrics()


async def verify_token(request: Request):
    token = request.headers.get("X-API-Token") or request.query_params.get("token")
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


class ControlUpdate(BaseModel):
    auto_trading: Optional[bool] = Field(
        default=None, description="Enable or disable live trade execution automation."
    )
    auto_retraining: Optional[bool] = Field(
        default=None, description="Toggle background model retraining."
    )
    shadow_mode: Optional[bool] = Field(
        default=None, description="Run the bot in continuous shadow testing mode."
    )
    emergency_stop: Optional[bool] = Field(
        default=None, description="Immediately halt live trading operations."
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/controls")
async def get_controls(_: None = Depends(verify_token)):
    return _control_payload()


@app.post("/controls")
async def update_controls(update: ControlUpdate, _: None = Depends(verify_token)):
    payload = update.model_dump(exclude_none=True)
    async with control_lock:
        state = _apply_control_update(payload)
    await _broadcast_controls(state)
    return state


@app.get("/shadow_status")
async def shadow_status(_: None = Depends(verify_token)):
    return get_shadow_status()


@app.get("/trades")
async def get_trades(_: None = Depends(verify_token)):
    return trades


@app.get("/metrics")
async def get_metrics(_: None = Depends(verify_token)):
    return get_aggregated_metrics()


@app.get("/decisions")
async def get_decisions(_: None = Depends(verify_token)):
    return decisions


@app.get("/training_progress")
async def get_training(_: None = Depends(verify_token)):
    return training_progress


def _auth_ws(ws: WebSocket) -> bool:
    token = ws.query_params.get("token")
    return not API_TOKEN or token == API_TOKEN


async def _ws_handler(
    ws: WebSocket,
    store: List[dict],
    connections: Set[WebSocket],
    on_message: Callable[[dict], None] | None = None,
):
    if not _auth_ws(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    connections.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                store.append(payload)
                if on_message:
                    on_message(payload)
            except Exception:
                payload = data
            # broadcast
            for conn in list(connections):
                if conn is not ws:
                    try:
                        await conn.send_text(data)
                    except Exception:
                        connections.discard(conn)
    except WebSocketDisconnect:
        pass
    finally:
        connections.discard(ws)


@app.websocket("/ws/trades")
async def ws_trades(ws: WebSocket):
    await _ws_handler(ws, trades, trade_connections)


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await _ws_handler(ws, metrics_store, metric_connections, _process_metric_payload)


@app.websocket("/ws/decisions")
async def ws_decisions(ws: WebSocket):
    await _ws_handler(ws, decisions, decision_connections)


@app.websocket("/ws/training_progress")
async def ws_training(ws: WebSocket):
    await _ws_handler(ws, training_progress, training_connections)


@app.websocket("/ws/controls")
async def ws_controls(ws: WebSocket):
    if not _auth_ws(ws):
        await ws.close(code=1008)
        return

    await ws.accept()
    control_connections.add(ws)
    try:
        await ws.send_text(json.dumps(_control_payload()))
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                update = ControlUpdate(**payload)
            except Exception:
                continue
            async with control_lock:
                state = _apply_control_update(update.model_dump(exclude_none=True))
            await _broadcast_controls(state)
    except WebSocketDisconnect:
        pass
    finally:
        control_connections.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
