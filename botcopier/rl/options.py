import numpy as np
from typing import List

try:  # pragma: no cover - optional dependency
    from gym import Env, spaces  # type: ignore
except Exception:  # pragma: no cover - gymnasium fallback
    from gymnasium import Env, spaces  # type: ignore

from .skills import (
    EntrySkill,
    ExitSkill,
    RiskSkill,
    SkillPolicy,
    default_skills,
)


class OptionTradeEnv(Env):
    """Environment where actions select high-level skills."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        skills: List[SkillPolicy] | None = None,
    ) -> None:
        super().__init__()
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=int)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.skills = skills or default_skills()
        self.action_space = spaces.Discrete(len(self.skills))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.states.shape[1],),
            dtype=np.float32,
        )
        self.idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        del seed, options
        self.idx = 0
        return self.states[self.idx], {}

    def step(self, option: int):  # type: ignore[override]
        option = int(option)
        skill = self.skills[option]
        correct = skill.act(self.states[self.idx]) == self.actions[self.idx]
        reward = float(self.rewards[self.idx]) if correct else 0.0
        self.idx += 1
        done = self.idx >= len(self.states)
        obs = self.states[self.idx] if not done else self.states[-1]
        return obs, reward, done, False, {}


def evaluate_option_policy(model, env: OptionTradeEnv) -> float:
    """Evaluate ``model`` on ``env`` returning the total reward."""

    obs, _ = env.reset()
    total = 0.0
    for _ in range(len(env.states)):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(int(action))
        total += float(r)
        if done:
            break
    return total


__all__ = [
    "OptionTradeEnv",
    "evaluate_option_policy",
    "SkillPolicy",
    "EntrySkill",
    "ExitSkill",
    "RiskSkill",
    "default_skills",
]
