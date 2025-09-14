"""Lightweight Reptile meta learning utilities.

This module provides a tiny implementation of the Reptile algorithm for a
logistic regression style model.  It mirrors the helper used by the
``scripts.meta_adapt`` module but lives in the ``meta`` package so that tests
and other components can import it without going through the scripts stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


def _logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of logistic loss for weights ``w``."""

    preds = 1.0 / (1.0 + np.exp(-(X @ w)))
    return X.T @ (preds - y) / len(y)


def evaluate(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return classification accuracy for ``w`` on ``(X, y)``."""

    preds = 1.0 / (1.0 + np.exp(-(X @ w))) > 0.5
    return float(np.mean(preds == y))


@dataclass
class ReptileMetaLearner:
    """Simple Reptile meta learner for dense logistic models."""

    dim: int
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        if self.weights is None:
            self.weights = np.zeros(self.dim, dtype=float)

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

    def adapt(
        self, X: np.ndarray, y: np.ndarray, inner_steps: int = 5, inner_lr: float = 0.1
    ) -> np.ndarray:
        """Return task adapted weights for new data ``(X, y)``."""

        w = self.weights.copy()
        for _ in range(inner_steps):
            w -= inner_lr * _logistic_grad(w, X, y)
        return w


__all__ = ["ReptileMetaLearner", "_logistic_grad", "evaluate"]

