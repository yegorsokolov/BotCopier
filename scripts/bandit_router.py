#!/usr/bin/env python3
"""Simple bandit router using Thompson Sampling or UCB.

The router listens on a TCP port and responds to simple commands:

    CHOOSE\n         -> returns an integer model index on its own line
    REWARD <idx> <reward>\n
         -> updates the specified model with a reward (1 for win, 0 for loss)

State is persisted to ``bandit_state.json`` by default so exploration can be
rolled back by deleting the file.
"""
import argparse
import json
import math
import os
import random
import socketserver
import threading
from typing import List


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
            return
        with self.lock:
            self.total[idx] += 1
            if reward > 0:
                self.wins[idx] += 1
            self._save_state()


class BanditTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:  # type: ignore[override]
        data = self.rfile.readline().decode().strip().split()
        if not data:
            return
        cmd = data[0].upper()
        if cmd == "CHOOSE":
            idx = self.server.router.choose()  # type: ignore[attr-defined]
            self.wfile.write(f"{idx}\n".encode())
        elif cmd == "REWARD" and len(data) >= 3:
            try:
                idx = int(data[1])
                reward = float(data[2])
            except ValueError:
                self.wfile.write(b"ERR\n")
                return
            self.server.router.update(idx, reward)  # type: ignore[attr-defined]
            self.wfile.write(b"OK\n")
        else:
            self.wfile.write(b"ERR\n")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, router):
        super().__init__(server_address, RequestHandlerClass)
        self.router = router


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit model router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--models", type=int, default=1,
                        help="number of models to route between")
    parser.add_argument("--method", choices=["thompson", "ucb"], default="thompson")
    parser.add_argument("--state-file", default="bandit_state.json")
    args = parser.parse_args()

    router = BanditRouter(args.models, args.method, args.state_file)
    with ThreadedTCPServer((args.host, args.port), BanditTCPHandler, router) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
