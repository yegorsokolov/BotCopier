from __future__ import annotations

"""Policy-gradient controller for AutoML feature/model selection."""

import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

Action = Tuple[Tuple[str, ...], str]


class AutoMLController:
    """Simple REINFORCE-style policy gradient controller.

    Parameters
    ----------
    features:
        Iterable of available feature names.
    models:
        Mapping of model name to an integer representing its relative
        complexity. Higher values imply greater complexity penalty.
    model_path:
        File used to persist the controller's policy and best action.
    reuse:
        When ``True`` the controller attempts to load an existing policy from
        ``model_path``. When ``False`` any existing policy is ignored.
    """

    def __init__(
        self,
        features: Iterable[str],
        models: Dict[str, int],
        model_path: str | os.PathLike[str] = "model.json",
        *,
        reuse: bool = True,
    ) -> None:
        self.features = list(features)
        self.models = models
        self.model_path = Path(model_path)
        self.action_space: List[Action] = []
        for r in range(1, len(self.features) + 1):
            for subset in itertools.combinations(self.features, r):
                for model in models:
                    self.action_space.append((subset, model))

        # policy parameters and reward estimates per action
        self.theta: Dict[str, float] = {self._key(a): 0.0 for a in self.action_space}
        self.avg_reward: Dict[str, float] = {
            self._key(a): 0.0 for a in self.action_space
        }
        self.counts: Dict[str, int] = {self._key(a): 0 for a in self.action_space}

        if reuse:
            self._load()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the learned policy."""
        for k in self.theta:
            self.theta[k] = 0.0
            self.avg_reward[k] = 0.0
            self.counts[k] = 0

    def _key(self, action: Action) -> str:
        subset, model = action
        subset_key = ",".join(sorted(subset))
        return f"{subset_key}|{model}"

    def _decode(self, key: str) -> Action:
        subset_key, model = key.split("|")
        subset = tuple(filter(None, subset_key.split(",")))
        return subset, model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text())
        except Exception:
            return
        sec = data.get("automl_controller", {})
        theta = sec.get("policy")
        rewards = sec.get("avg_reward")
        counts = sec.get("counts")
        if theta:
            for k, v in theta.items():
                if k in self.theta:
                    self.theta[k] = float(v)
        if rewards:
            for k, v in rewards.items():
                if k in self.avg_reward:
                    self.avg_reward[k] = float(v)
        if counts:
            for k, v in counts.items():
                if k in self.counts:
                    self.counts[k] = int(v)

    def _save(self) -> None:
        try:
            data = json.loads(self.model_path.read_text())
        except Exception:
            data = {}
        sec = data.setdefault("automl_controller", {})
        sec["policy"] = self.theta
        sec["avg_reward"] = self.avg_reward
        sec["counts"] = self.counts
        best = self.select_best()
        if best is not None:
            sec["best_action"] = {"features": list(best[0]), "model": best[1]}
        with self.model_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Policy logic
    # ------------------------------------------------------------------
    def _probs(self) -> List[float]:
        weights = [self.theta[self._key(a)] for a in self.action_space]
        max_w = max(weights)
        exps = [math.exp(w - max_w) for w in weights]
        s = sum(exps)
        return [e / s for e in exps]

    def sample_action(self) -> Tuple[Action, List[float]]:
        """Sample an action according to the current policy."""
        probs = self._probs()
        action = random.choices(self.action_space, probs)[0]
        return action, probs

    def update(self, action: Action, reward: float, alpha: float = 0.1) -> None:
        """Update policy parameters for ``action`` with ``reward``."""
        probs = self._probs()
        chosen = self.action_space.index(action)
        keys = [self._key(a) for a in self.action_space]
        for i, k in enumerate(keys):
            grad = (1.0 if i == chosen else 0.0) - probs[i]
            self.theta[k] += alpha * reward * grad
        k = keys[chosen]
        self.counts[k] += 1
        c = self.counts[k]
        self.avg_reward[k] += (reward - self.avg_reward[k]) / c
        self._save()

    def train(
        self,
        env: Callable[[Action], float | tuple[float, float]],
        episodes: int = 100,
        alpha: float = 0.1,
        penalty: float = 0.1,
        risk_limit: float | None = None,
    ) -> None:
        """Train the controller in ``env`` for ``episodes`` iterations."""
        for _ in range(episodes):
            action, _ = self.sample_action()
            result = env(action)
            if isinstance(result, tuple):
                profit, risk = result
            else:
                profit, risk = float(result), 0.0
            subset, model = action
            complexity = len(subset) + self.models[model]
            risk_pen = 0.0
            if risk_limit is not None and risk > risk_limit:
                risk_pen = risk - risk_limit
            reward = profit - penalty * complexity - risk_pen
            self.update(action, reward, alpha=alpha)

    def select_best(self) -> Action | None:
        if not self.avg_reward:
            return None
        key = max(self.avg_reward, key=self.avg_reward.get)
        return self._decode(key)
