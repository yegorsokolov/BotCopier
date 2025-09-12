#!/usr/bin/env python3
"""Weighted traffic router for candidate models.

The router forwards requests to multiple candidate models according to a
configurable traffic split.  Outcomes can be recorded for each model and the
router can be instructed to switch all traffic to the best performing model.

This utility is intentionally lightweight so it can be imported directly in
unit tests without requiring a running web service.  The interface mirrors the
``BanditRouter`` in ``bandit_router.py`` but focuses on deterministic traffic
splitting rather than adaptive bandit algorithms.
"""
from __future__ import annotations

import argparse
import random
from typing import Iterable, List, Sequence

from botcopier.utils.random import set_seed


class TrafficRouter:
    """Route requests between models using pre-defined traffic shares.

    Parameters
    ----------
    weights:
        Sequence of non-negative numbers defining the probability of routing to
        each model.  They do not need to sum to one; normalisation happens
        automatically.
    auto_switch:
        When ``True`` the router will automatically divert all traffic to the
        model with the highest observed win rate whenever outcomes are
        recorded via :meth:`update`.
    seed:
        Optional seed for deterministic behaviour during testing.
    """

    def __init__(
        self, weights: Sequence[float], auto_switch: bool = False, seed: int | None = None
    ) -> None:
        if not weights:
            raise ValueError("weights must contain at least one value")
        if seed is not None:
            set_seed(seed)
        self.weights: List[float] = [float(w) for w in weights]
        self.auto_switch = bool(auto_switch)
        self.total: List[int] = [0] * len(self.weights)
        self.wins: List[int] = [0] * len(self.weights)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _normalised_weights(self) -> List[float]:
        total = sum(max(w, 0.0) for w in self.weights)
        if total <= 0:
            return [1.0 / len(self.weights)] * len(self.weights)
        return [max(w, 0.0) / total for w in self.weights]

    def choose(self) -> int:
        """Return the index of the next model according to current weights."""
        probs = self._normalised_weights()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------
    def update(self, idx: int, reward: float) -> None:
        """Record the outcome for ``idx`` and optionally auto-switch traffic."""
        if idx < 0 or idx >= len(self.weights):
            return
        self.total[idx] += 1
        if reward > 0:
            self.wins[idx] += 1
        if self.auto_switch:
            self.switch_to_best()

    def best_model(self) -> int:
        """Return the model index with the highest observed win rate."""
        rates = [
            (self.wins[i] / self.total[i]) if self.total[i] > 0 else -1.0
            for i in range(len(self.weights))
        ]
        return max(range(len(self.weights)), key=lambda i: rates[i])

    def switch_to_best(self) -> int:
        """Redirect all traffic to the best performing model."""
        best = self.best_model()
        self.weights = [0.0] * len(self.weights)
        self.weights[best] = 1.0
        return best

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_weights(value: str) -> List[float]:
        return [float(x) for x in value.split(",") if x]


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Traffic splitting router")
    parser.add_argument(
        "--weights", default="1", help="Comma separated traffic weights, e.g. 0.8,0.2"
    )
    parser.add_argument(
        "--auto-switch", action="store_true", help="Automatically switch to best"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--switch-best",
        action="store_true",
        help="Immediately switch to the currently best model and exit",
    )
    parsed = parser.parse_args(args=args)

    router = TrafficRouter(
        TrafficRouter._parse_weights(parsed.weights),
        auto_switch=parsed.auto_switch,
        seed=parsed.seed,
    )
    if parsed.switch_best:
        router.switch_to_best()
        print(router.weights)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
