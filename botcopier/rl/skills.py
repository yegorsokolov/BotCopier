import numpy as np
from pathlib import Path
from typing import List

class SkillPolicy:
    """Deterministic low-level policy representing a discrete action."""

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def act(self, state: np.ndarray) -> int:  # pragma: no cover - trivial
        del state
        return self.action


class EntrySkill(SkillPolicy):
    """Skill representing order entry."""

    def __init__(self) -> None:
        super().__init__(0)


class ExitSkill(SkillPolicy):
    """Skill representing exiting a position."""

    def __init__(self) -> None:
        super().__init__(1)


class RiskSkill(SkillPolicy):
    """Skill representing risk management actions."""

    def __init__(self) -> None:
        super().__init__(2)


def default_skills() -> List[SkillPolicy]:
    """Return the default set of skills: entry, exit and risk control."""

    return [EntrySkill(), ExitSkill(), RiskSkill()]


class HighLevelPolicy:
    """Simple linear policy selecting among provided skills.

    The policy learns a weight matrix mapping observations to skill scores.
    Updates use a basic TD-style rule which is sufficient for the unit tests
    and keeps the implementation lightweight.
    """

    def __init__(
        self,
        n_features: int,
        skills: List[SkillPolicy],
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        self.weights = np.zeros((n_features, len(skills)), dtype=np.float32)
        self.skills = skills
        self.lr = learning_rate
        self.gamma = gamma

    def predict(self, state: np.ndarray) -> int:
        """Return the index of the best skill for ``state``."""

        q_values = state @ self.weights
        return int(np.argmax(q_values))

    def update(self, state: np.ndarray, action: int, reward: float) -> None:
        """Update policy weights given ``state``, ``action`` and ``reward``."""

        q = float(state @ self.weights[:, action])
        td_error = reward - q
        self.weights[:, action] += self.lr * td_error * state

    def save(self, path: Path) -> None:
        """Persist weights to ``path``."""

        np.save(path, self.weights)

    @classmethod
    def load(
        cls,
        path: Path,
        skills: List[SkillPolicy],
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ) -> "HighLevelPolicy":
        """Load a policy from ``path``."""

        weights = np.load(path)
        policy = cls(weights.shape[0], skills, learning_rate, gamma)
        policy.weights = weights
        return policy


__all__ = [
    "SkillPolicy",
    "EntrySkill",
    "ExitSkill",
    "RiskSkill",
    "default_skills",
    "HighLevelPolicy",
]
