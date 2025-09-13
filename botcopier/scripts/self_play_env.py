"""Adversarial market simulator for simple self-play training.

This module provides a tiny synthetic environment where two agents interact:
``Trader`` and ``Perturber``.  The trader chooses whether to buy, sell or hold
an asset while the perturbation agent injects shocks into the price dynamics in
order to reduce the trader's profit.  The game is zero-sum – the perturbation
agent receives the negative of the trader's reward.

The implementation purposely keeps the dynamics minimal so tests can train the
agents in only a few iterations.  Utilities for training both agents via
alternating Q-learning updates and evaluating the learnt policies are included
to make it easy to integrate into higher level training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayEnv:
    """Very small zero-sum market game.

    Parameters
    ----------
    steps:
        Number of interaction steps in an episode.  In this simplified setting
        the state does not change between steps so ``steps`` mainly controls the
        amount of experience gathered during training.
    drift:
        Base drift of the price process without adversarial interference.
    volatility:
        Standard deviation of Gaussian noise added each step.
    perturb_scale:
        Magnitude of price shock applied by the perturbation agent when it
        chooses ``+1`` or ``-1``.
    seed:
        Optional random seed for reproducibility.
    """

    steps: int = 1
    drift: float = 0.01
    volatility: float = 0.0
    perturb_scale: float = 0.05
    seed: int | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> float:
        """Reset the simulator and return the initial observation."""

        self.t = 0
        return 0.0  # observation is unused but kept for API symmetry

    def step(
        self, trade_action: int, perturb_action: int
    ) -> Tuple[float, float, float, bool, dict]:
        """Advance the simulation by one step.

        Parameters
        ----------
        trade_action:
            Trader decision: ``-1`` (sell), ``0`` (hold) or ``1`` (buy).
        perturb_action:
            Adversary decision: ``-1`` (downward shock), ``0`` (none) or
            ``1`` (upward shock).
        """

        noise = self.rng.standard_normal() * self.volatility
        price_change = self.drift + noise + self.perturb_scale * perturb_action
        reward_trader = trade_action * price_change
        reward_perturb = -reward_trader
        self.t += 1
        done = self.t >= self.steps
        # observation is irrelevant for the simple strategies used in tests
        return 0.0, reward_trader, reward_perturb, done, {}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _epsilon_greedy(q: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """Return an index following an epsilon-greedy policy."""

    if rng.random() < eps:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


def train_trader_only(
    env: SelfPlayEnv,
    episodes: int = 100,
    lr: float = 0.1,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Train only the trader in the absence of an adversary.

    A simple Q-learning update is used where the state space collapses to a
    single state.  The learnt Q-values therefore directly correspond to the
    expected reward of taking each action.
    """

    q_trader = np.zeros(3, dtype=float)  # actions: sell, hold, buy
    for _ in range(episodes):
        env.reset()
        idx = _epsilon_greedy(q_trader, epsilon, env.rng)
        trade_action = idx - 1
        _, reward, _, _, _ = env.step(trade_action, 0)
        q_trader[idx] += lr * (reward - q_trader[idx])
    return q_trader


def train_self_play(
    env: SelfPlayEnv,
    episodes: int = 100,
    lr: float = 0.1,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train trader and perturbation agents via alternating updates."""

    trader_q = np.zeros(3, dtype=float)
    pert_q = np.zeros(3, dtype=float)
    for ep in range(episodes):
        env.reset()
        t_idx = _epsilon_greedy(trader_q, epsilon, env.rng)
        p_idx = _epsilon_greedy(pert_q, epsilon, env.rng)
        t_act = t_idx - 1
        p_act = p_idx - 1
        _, r_t, r_p, _, _ = env.step(t_act, p_act)
        if ep % 2 == 0:  # update trader on even episodes
            trader_q[t_idx] += lr * (r_t - trader_q[t_idx])
        else:  # update perturbation agent on odd episodes
            pert_q[p_idx] += lr * (r_p - pert_q[p_idx])
    return trader_q, pert_q


def evaluate(
    env: SelfPlayEnv,
    trader_q: np.ndarray,
    perturb_action: int,
    episodes: int = 100,
) -> float:
    """Evaluate trader policy against a fixed adversary action."""

    total = 0.0
    for _ in range(episodes):
        env.reset()
        idx = int(np.argmax(trader_q))
        trade_action = idx - 1
        _, reward, _, _, _ = env.step(trade_action, perturb_action)
        total += reward
    return total / episodes


__all__ = [
    "SelfPlayEnv",
    "train_trader_only",
    "train_self_play",
    "evaluate",
]
