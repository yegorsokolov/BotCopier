#!/usr/bin/env python3
"""Simple bandit router using Thompson Sampling or UCB.

The router exposes a tiny HTTP API:

* ``GET /choose``   – returns the next model index to use
* ``POST /reward``  – update the chosen model with binary win/loss feedback

State is persisted to ``bandit_state.json`` by default so exploration can be
reset by deleting the file.
"""
import argparse
import json
import math
import os
import random
import threading
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from botcopier.metrics import (
    ERROR_COUNTER,
    latest_metrics,
    observe_latency,
    start_metrics_server,
)
from botcopier.utils.random import set_seed


class BanditRouter:
    """Route requests between multiple models using a bandit algorithm."""

    def __init__(self, models: int, method: str, state_file: str) -> None:
        self.models = models
        self.method = method
        self.state_file = state_file
        self.lock = threading.Lock()
        # total[i] = total pulls for model i, wins[i] = winning pulls
        self.total: List[int] = [0] * models
        self.wins: List[int] = [0] * models
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.total = data.get("total", self.total)
                self.wins = data.get("wins", self.wins)
            except Exception:
                pass

    def _save_state(self) -> None:
        tmp = {"total": self.total, "wins": self.wins}
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(tmp, fh)

    # ------------------------------------------------------------------
    # Bandit algorithms
    # ------------------------------------------------------------------
    def _choose_thompson(self) -> int:
        best = 0
        best_sample = -1.0
        for i in range(self.models):
            a = self.wins[i] + 1
            b = (self.total[i] - self.wins[i]) + 1
            sample = random.betavariate(a, b)
            if sample > best_sample:
                best_sample = sample
                best = i
        return best

    def _choose_ucb(self) -> int:
        t = sum(self.total) + 1
        scores = []
        for i in range(self.models):
            if self.total[i] == 0:
                scores.append(float("inf"))
            else:
                mean = self.wins[i] / self.total[i]
                bonus = math.sqrt(2 * math.log(t) / self.total[i])
                scores.append(mean + bonus)
        return max(range(self.models), key=scores.__getitem__)

    def choose(self) -> int:
        with self.lock:
            if self.method == "ucb":
                return self._choose_ucb()
            return self._choose_thompson()

    def update(self, idx: int, reward: float) -> None:
        if idx < 0 or idx >= self.models:
            ERROR_COUNTER.labels(type="bandit_reward").inc()
            return
        with self.lock:
            self.total[idx] += 1
            if reward > 0:
                self.wins[idx] += 1
            self._save_state()


class Reward(BaseModel):
    model: int
    reward: float


def create_app(router: BanditRouter) -> FastAPI:
    app = FastAPI()

    @app.get("/choose")
    def choose() -> dict[str, int]:
        with observe_latency("bandit_choose"):
            return {"model": router.choose()}

    @app.post("/reward")
    def reward(update: Reward) -> dict[str, str]:
        with observe_latency("bandit_reward"):
            router.update(update.model, update.reward)
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        payload, content_type = latest_metrics()
        return Response(content=payload, media_type=content_type)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit model router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--models", type=int, default=1, help="number of models to route between"
    )
    parser.add_argument("--method", choices=["thompson", "ucb"], default="thompson")
    parser.add_argument("--state-file", default="bandit_state.json")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9101,
        help="Prometheus metrics port",
    )
    args = parser.parse_args()

    set_seed(args.random_seed)
    router = BanditRouter(args.models, args.method, args.state_file)
    app = create_app(router)
    start_metrics_server(args.metrics_port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
