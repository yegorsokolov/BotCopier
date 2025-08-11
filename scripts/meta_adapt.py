#!/usr/bin/env python3
"""Meta-adaptation service and utilities.

This module implements a very small meta learning example using the
Reptile algorithm for logistic regression models.  It can be used as a
stand‑alone training script, a server that serves adapted weights over a
plain TCP/JSON protocol, and also provides helper functions used in unit
tests.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of logistic loss for weights ``w``."""
    preds = _sigmoid(X @ w)
    return X.T @ (preds - y) / len(y)


def evaluate(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return accuracy for weights ``w`` on dataset ``(X, y)``."""
    preds = _sigmoid(X @ w) > 0.5
    return float(np.mean(preds == y))


def log_adaptation(
    old_w: Sequence[float],
    new_w: Sequence[float],
    regime_id: int,
    log_path: str,
) -> None:
    """Append an adaptation event to ``log_path``.

    Each line contains: ``timestamp,regime_id,old_weights,new_weights`` with
    weights written as space separated floats.
    """
    ts = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{ts},{regime_id},{' '.join(map(str, old_w))},{' '.join(map(str, new_w))}\n"
        )


# ---------------------------------------------------------------------------
# Reptile meta learner
# ---------------------------------------------------------------------------


@dataclass
class ReptileMetaLearner:
    """Simple Reptile meta learner for logistic regression."""

    dim: int
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros(self.dim, dtype=float)

    # --------------------------- training ---------------------------------
    def train(
        self,
        sessions: Iterable[Tuple[np.ndarray, np.ndarray]],
        inner_steps: int = 5,
        inner_lr: float = 0.1,
        meta_lr: float = 0.1,
    ) -> None:
        """Train meta weights from ``sessions`` using the Reptile algorithm."""
        for X, y in sessions:
            w = self.weights.copy()
            for _ in range(inner_steps):
                w -= inner_lr * _logistic_grad(w, X, y)
            self.weights += meta_lr * (w - self.weights)

    # --------------------------- adaptation ------------------------------
    def adapt(
        self, X: np.ndarray, y: np.ndarray, inner_steps: int = 5, inner_lr: float = 0.1
    ) -> np.ndarray:
        """Return weights adapted to a new session ``(X, y)``."""
        w = self.weights.copy()
        for _ in range(inner_steps):
            w -= inner_lr * _logistic_grad(w, X, y)
        return w


# ---------------------------------------------------------------------------
# Network service
# ---------------------------------------------------------------------------

def _serve(meta: ReptileMetaLearner, port: int, log_path: str) -> None:
    """Serve adapted weights over a simple TCP socket.

    The client should send a JSON object ``{"regime": int, "X": [[..]],
    "y": [..]}``.  The response will contain a JSON object with
    ``coefficients`` and ``intercept`` fields.  Only coefficients are
    currently adapted; the intercept is always ``0`` as the toy learner does
    not model it separately.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", port))
    sock.listen()

    while True:
        conn, _ = sock.accept()
        data = conn.recv(1 << 20).decode("utf-8")
        try:
            req = json.loads(data)
            regime = int(req.get("regime", 0))
            X = np.array(req.get("X", []), dtype=float)
            y = np.array(req.get("y", []), dtype=float)
            new_w = meta.adapt(X, y)
            log_adaptation(meta.weights, new_w, regime, log_path)
            meta.weights = new_w
            resp = json.dumps({"coefficients": new_w.tolist(), "intercept": 0.0})
            conn.sendall(resp.encode("utf-8"))
        except Exception:
            # Ignore malformed requests
            conn.sendall(b"{}")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------

def _load_sessions(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    sessions_raw = json.loads(Path(path).read_text())
    sessions = []
    for sess in sessions_raw:
        X = np.array(sess["X"], dtype=float)
        y = np.array(sess["y"], dtype=float)
        sessions.append((X, y))
    return sessions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meta adaptation utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train meta learner from sessions")
    train_p.add_argument("sessions", help="JSON file with training sessions")
    train_p.add_argument("out", help="Output JSON file for meta weights")
    train_p.add_argument("--inner-steps", type=int, default=5)
    train_p.add_argument("--inner-lr", type=float, default=0.1)
    train_p.add_argument("--meta-lr", type=float, default=0.1)

    serve_p = sub.add_parser("serve", help="Serve adapted weights over TCP")
    serve_p.add_argument("model", help="JSON file with meta weights")
    serve_p.add_argument("--port", type=int, default=9200)
    serve_p.add_argument("--log", default="adaptations.csv")

    adapt_p = sub.add_parser("adapt", help="Adapt weights for given data")
    adapt_p.add_argument("model", help="JSON file with meta weights")
    adapt_p.add_argument("data", help="JSON file with X and y arrays")
    adapt_p.add_argument("--out", help="Optional output file for adapted weights")
    adapt_p.add_argument("--log", default="adaptations.csv")
    adapt_p.add_argument("--inner-steps", type=int, default=5)
    adapt_p.add_argument("--inner-lr", type=float, default=0.1)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        sessions = _load_sessions(Path(args.sessions))
        dim = sessions[0][0].shape[1]
        meta = ReptileMetaLearner(dim)
        meta.train(sessions, inner_steps=args.inner_steps, inner_lr=args.inner_lr, meta_lr=args.meta_lr)
        out = {"weights": meta.weights.tolist()}
        Path(args.out).write_text(json.dumps(out))
        return 0

    if args.cmd == "serve":
        weights = np.array(json.loads(Path(args.model).read_text())["weights"], dtype=float)
        meta = ReptileMetaLearner(len(weights), weights)
        _serve(meta, args.port, args.log)
        return 0

    if args.cmd == "adapt":
        weights = np.array(json.loads(Path(args.model).read_text())["weights"], dtype=float)
        data = json.loads(Path(args.data).read_text())
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)
        regime = int(data.get("regime", 0))
        meta = ReptileMetaLearner(len(weights), weights)
        new_w = meta.adapt(X, y, inner_steps=args.inner_steps, inner_lr=args.inner_lr)
        log_adaptation(weights, new_w, regime, args.log)
        out = {"weights": new_w.tolist()}
        if args.out:
            Path(args.out).write_text(json.dumps(out))
        else:
            print(json.dumps(out))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
