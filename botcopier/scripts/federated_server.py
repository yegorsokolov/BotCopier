#!/usr/bin/env python3
"""Simple federated averaging server."""

from __future__ import annotations

import argparse
from threading import Lock
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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
    with state["lock"]:
        state["updates"].append(update)
        if len(state["updates"]) >= state["expected"]:
            ws = [u.weights for u in state["updates"]]
            state["weights"] = [sum(col) / len(col) for col in zip(*ws)]
            inters = [u.intercept for u in state["updates"] if u.intercept is not None]
            state["intercept"] = (
                sum(inters) / len(inters) if inters else None
            )
            state["updates"] = []
    return {"weights": state["weights"], "intercept": state["intercept"]}


@app.get("/weights")
def weights():
    return {"weights": state["weights"], "intercept": state["intercept"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Federated averaging server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--clients", type=int, default=2, help="number of clients to wait for"
    )
    args = p.parse_args()
    state["expected"] = args.clients
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

