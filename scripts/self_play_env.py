#!/usr/bin/env python3
"""Gym environment generating synthetic price paths and simple order-book dynamics."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - gym optional dependency
    from gym import Env, spaces  # type: ignore
except Exception:  # pragma: no cover - gymnasium fallback
    from gymnasium import Env, spaces  # type: ignore


class SelfPlayEnv(Env):
    """Synthetic environment for training agents without real market data.

    The environment simulates a rudimentary order book around a mid price
    following a random walk with optional drift and volatility.  Actions are
    ``0`` (hold), ``1`` (buy) and ``2`` (sell).  Rewards are the mark-to-market
    PnL after transaction costs based on the current spread.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        steps: int = 100,
        drift: float = 0.0,
        volatility: float = 1.0,
        spread: float = 0.0002,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.spread = float(spread)
        self.rng = np.random.default_rng(seed)

        # action: 0 hold, 1 buy, 2 sell
        self.action_space = spaces.Discrete(3)

        high = np.finfo(np.float32).max
        # observation: [mid_price, bid, ask, position]
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(4,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        bid = self.price - self.spread / 2.0
        ask = self.price + self.spread / 2.0
        return np.array([self.price, bid, ask, self.position], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.price = 100.0
        self.position = 0.0
        self.pnl = 0.0
        return self._get_obs(), {}

    def step(self, action: int):  # type: ignore[override]
        prev_price = self.price
        self.price += self.drift + self.volatility * self.rng.standard_normal()
        reward = 0.0
        bid = self.price - self.spread / 2.0
        ask = self.price + self.spread / 2.0

        if action == 1:  # buy
            self.position += 1.0
            reward -= ask * self.spread
        elif action == 2:  # sell
            self.position -= 1.0
            reward -= bid * self.spread

        reward += self.position * (self.price - prev_price)
        self.pnl += reward
        self.t += 1
        done = self.t >= self.steps
        obs = np.array([self.price, bid, ask, self.position], dtype=np.float32)
        info = {"pnl": float(self.pnl)}
        return obs, float(reward), done, False, info

