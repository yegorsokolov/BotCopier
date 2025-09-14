"""Minimal adversarial market environment for self-play training.

Two agents interact in this toy environment:

* **Trader** – chooses to buy (1), sell (-1) or hold (0).
* **Perturbator** – injects a price shock of ``perturb_scale`` either
  up (1) or down (-1) to reduce the trader's profit.  ``0`` represents
  no shock.  The perturbator's reward is the negative of the trader's
  reward making the game zero-sum.

The implementation purposefully keeps the state space trivial so that
unit tests can train the agents in a few iterations.  Training utilities
implement alternating Q-learning updates with basic logging of rewards
and strategy resilience.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayEnv:
    """Tiny zero-sum market simulator.

    Parameters
    ----------
    steps:
        Number of steps in an episode.  The observation does not change
        so this mainly controls the amount of experience gathered during
        training.
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
        """Advance the simulation by one step."""

        noise = self.rng.standard_normal() * self.volatility
        price_change = self.drift + noise + self.perturb_scale * perturb_action
        reward_trader = trade_action * price_change
        reward_perturb = -reward_trader
        self.t += 1
        done = self.t >= self.steps
        # observation is irrelevant for the strategies used in tests
        return 0.0, reward_trader, reward_perturb, done, {}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _epsilon_greedy(q: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """Select an action index following an epsilon-greedy policy."""

    if rng.random() < eps:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


def train_trader_only(
    env: SelfPlayEnv,
    episodes: int = 100,
    lr: float = 0.1,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Train only the trader against historical dynamics.

    The perturbation agent always chooses ``0`` (no shock).  A single state
    Q-learning update is sufficient as the observation is constant.
    """

    q_trader = np.zeros(3, dtype=float)  # sell, hold, buy
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
    """Train trader and perturbator via alternating updates.

    Rewards and the trader's resilience (profit under a constant downward
    shock) are logged every 10 episodes.
    """

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
        if ep % 10 == 0:
            resilience = evaluate(env, trader_q, perturb_action=-1, episodes=10)
            logger.debug(
                {
                    "episode": ep,
                    "reward_trader": float(r_t),
                    "reward_perturb": float(r_p),
                    "resilience": float(resilience),
                }
            )
    return trader_q, pert_q


def evaluate(
    env: SelfPlayEnv,
    trader_q: np.ndarray,
    perturb_action: int,
    episodes: int = 100,
) -> float:
    """Evaluate a trader policy against a fixed perturbation action."""

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
