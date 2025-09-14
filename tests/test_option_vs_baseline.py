from __future__ import annotations

from pathlib import Path

import pytest

from tests import HAS_NUMPY, HAS_SB3

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

try:  # pragma: no cover - gymnasium compatibility
    from gym import Env, spaces  # type: ignore
except Exception:  # pragma: no cover
    try:
        from gymnasium import Env, spaces  # type: ignore
    except Exception:  # pragma: no cover - provide dummies when gym not present
        Env = object  # type: ignore
        spaces = None  # type: ignore

if HAS_SB3 and HAS_NUMPY:
    from botcopier.rl.options import OptionTradeEnv, evaluate_option_policy
    from botcopier.rl.skills import default_skills
    from scripts.train_rl_agent import train_options
    from stable_baselines3 import PPO

pytestmark = pytest.mark.skipif(
    not (HAS_SB3 and HAS_NUMPY), reason="stable-baselines3 or numpy not installed"
)


def _build_dataset():
    states = []
    actions = []
    rewards = []
    for _ in range(10):
        states.append([1.0, 0.0, 0.0])
        actions.append(0)
        rewards.append(1.0)
    for _ in range(10):
        states.append([0.0, 1.0, 0.0])
        actions.append(1)
        rewards.append(1.0)
    for _ in range(10):
        states.append([0.0, 0.0, 1.0])
        actions.append(2)
        rewards.append(1.0)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=int),
        np.array(rewards, dtype=float),
    )


class TradeEnv(Env):
    """Environment where actions correspond directly to trades."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        super().__init__()
        self.states = states.astype(np.float32)
        self.actions = actions.astype(int)
        self.rewards = rewards.astype(np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.states.shape[1],), dtype=np.float32
        )
        self.idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        del seed, options
        self.idx = 0
        return self.states[self.idx], {}

    def step(self, action: int):  # type: ignore[override]
        action = int(action)
        correct = action == self.actions[self.idx]
        reward = float(self.rewards[self.idx]) if correct else 0.0
        self.idx += 1
        done = self.idx >= len(self.states)
        obs = self.states[self.idx] if not done else self.states[-1]
        return obs, reward, done, False, {}


def _evaluate_direct(model, env: TradeEnv) -> float:
    obs, _ = env.reset()
    total = 0.0
    for _ in range(len(env.states)):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(int(action))
        total += float(r)
        if done:
            break
    return total


def test_option_policy_beats_baseline(tmp_path: Path) -> None:
    states, actions, rewards = _build_dataset()

    base_env = TradeEnv(states, actions, rewards)
    baseline = PPO("MlpPolicy", base_env, learning_rate=0.1, gamma=0.99, seed=0, verbose=0)
    baseline_reward = _evaluate_direct(baseline, base_env)

    info = train_options(
        states,
        actions,
        rewards,
        tmp_path,
        training_steps=200,
        learning_rate=0.1,
    )
    model = PPO.load(str(tmp_path / info["option_weights_file"]))
    opt_env = OptionTradeEnv(states, actions, rewards, default_skills())
    option_reward = evaluate_option_policy(model, opt_env)

    assert option_reward > baseline_reward
