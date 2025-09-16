#!/usr/bin/env python3
"""Simple federated averaging server."""

from __future__ import annotations

import argparse
from threading import Lock
from typing import List

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from botcopier.metrics import (
    latest_metrics,
    observe_latency,
    start_metrics_server,
)


app = FastAPI()


class ModelUpdate(BaseModel):
    weights: List[float]
    intercept: float | None = None


state = {
    "updates": [],
    "weights": None,
    "intercept": None,
    "expected": 1,
    "lock": Lock(),
}


@app.post("/update")
def update(update: ModelUpdate):
    with observe_latency("federated_update"):
        with state["lock"]:
            state["updates"].append(update)
            if len(state["updates"]) >= state["expected"]:
                ws = [u.weights for u in state["updates"]]
                state["weights"] = [sum(col) / len(col) for col in zip(*ws)]
                inters = [
                    u.intercept for u in state["updates"] if u.intercept is not None
                ]
                state["intercept"] = (
                    sum(inters) / len(inters) if inters else None
                )
                state["updates"] = []
    return {"weights": state["weights"], "intercept": state["intercept"]}


@app.get("/weights")
def weights():
    with observe_latency("federated_weights"):
        return {"weights": state["weights"], "intercept": state["intercept"]}


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = latest_metrics()
    return Response(content=payload, media_type=content_type)


def main() -> None:
    p = argparse.ArgumentParser(description="Federated averaging server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--clients", type=int, default=2, help="number of clients to wait for"
    )
    p.add_argument(
        "--metrics-port",
        type=int,
        default=8005,
        help="Prometheus metrics port",
    )
    args = p.parse_args()
    state["expected"] = args.clients
    start_metrics_server(args.metrics_port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

