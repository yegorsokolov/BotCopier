import os
import json
from typing import List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import pyarrow.flight as flight

API_TOKEN = os.environ.get("DASHBOARD_API_TOKEN", "")
FLIGHT_URI = os.environ.get("FLIGHT_URI", "")

app = FastAPI(title="BotCopier Dashboard")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# In-memory stores
trades: List[dict] = []
metrics: List[dict] = []
decisions: List[dict] = []
training_progress: List[dict] = []

# Active WebSocket connections
trade_connections: Set[WebSocket] = set()
metric_connections: Set[WebSocket] = set()
decision_connections: Set[WebSocket] = set()
training_connections: Set[WebSocket] = set()

if FLIGHT_URI:
    try:
        client = flight.FlightClient(FLIGHT_URI)
        for name, store in (("trades", trades), ("metrics", metrics), ("decisions", decisions)):
            desc = flight.FlightDescriptor.for_path(name)
            info = client.get_flight_info(desc)
            reader = client.do_get(info.endpoints[0].ticket)
            table = reader.read_all()
            store.extend(table.to_pylist())
    except Exception:
        pass


async def verify_token(request: Request):
    token = request.headers.get("X-API-Token") or request.query_params.get("token")
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/trades")
async def get_trades(_: Request = Depends(verify_token)):
    return trades


@app.get("/metrics")
async def get_metrics(_: Request = Depends(verify_token)):
    return metrics


@app.get("/decisions")
async def get_decisions(_: Request = Depends(verify_token)):
    return decisions


@app.get("/training_progress")
async def get_training(_: Request = Depends(verify_token)):
    return training_progress


def _auth_ws(ws: WebSocket) -> bool:
    token = ws.query_params.get("token")
    return not API_TOKEN or token == API_TOKEN


async def _ws_handler(ws: WebSocket, store: List[dict], connections: Set[WebSocket]):
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
    await _ws_handler(ws, metrics, metric_connections)


@app.websocket("/ws/decisions")
async def ws_decisions(ws: WebSocket):
    await _ws_handler(ws, decisions, decision_connections)


@app.websocket("/ws/training_progress")
async def ws_training(ws: WebSocket):
    await _ws_handler(ws, training_progress, training_connections)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
