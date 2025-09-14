#!/usr/bin/env python3
"""Reinforcement-learning controller for AutoML.

This module implements a simple epsilon-greedy bandit agent that explores
combinations of feature subsets and model types.  Each action corresponds to a
specific subset of features and a chosen model type.  During training the agent
receives a reward based on the profit of the evaluated combination minus a
complexity penalty.  Learned policies are persisted to ``model.json`` so that
future runs can resume from previously discovered knowledge.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

Action = Tuple[Tuple[str, ...], str]


class AutoMLController:
    """Simple epsilon-greedy controller for feature/model selection.

    Parameters
    ----------
    features:
        Iterable of available feature names.
    models:
        Mapping of model name to an integer representing its relative
        complexity.  Higher numbers imply greater complexity penalty.
    model_path:
        Path to the ``model.json`` file used to persist policy information.
    """

    def __init__(
        self,
        features: Iterable[str],
        models: Dict[str, int],
        model_path: str | os.PathLike[str] = "model.json",
    ) -> None:
        self.features = list(features)
        self.models = models
        self.model_path = Path(model_path)
        self.action_space: List[Action] = []
        for r in range(1, len(self.features) + 1):
            for subset in itertools.combinations(self.features, r):
                for model in models:
                    self.action_space.append((subset, model))

        self.q_values: Dict[str, float] = {}
        self._data: Dict[str, object] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _key(self, action: Action) -> str:
        subset, model = action
        subset_key = ",".join(sorted(subset))
        return f"{subset_key}|{model}"

    def _decode_key(self, key: str) -> Action:
        subset_key, model = key.split("|")
        subset = tuple(filter(None, subset_key.split(",")))
        return subset, model

    def _load(self) -> None:
        if self.model_path.exists():
            with self.model_path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)
            self.q_values = self._data.get("automl_controller", {}).get("q_values", {})
        else:
            self._data = {}

    def _save(self) -> None:
        payload = self._data.setdefault("automl_controller", {})
        payload["q_values"] = self.q_values
        if self.q_values:
            best_key = max(self.q_values, key=self.q_values.get)
            subset, model = self._decode_key(best_key)
            payload["best_action"] = {
                "features": list(subset),
                "model": model,
            }
        else:
            payload["best_action"] = None
        with self.model_path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # RL logic
    # ------------------------------------------------------------------
    def choose_action(self, epsilon: float) -> Action:
        if random.random() < epsilon or not self.q_values:
            return random.choice(self.action_space)
        best_key = max(self.q_values, key=self.q_values.get)
        return self._decode_key(best_key)

    def train(
        self,
        env: Callable[[Action], float | tuple[float, float]],
        episodes: int = 100,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        penalty: float = 0.1,
        risk_limit: float | None = None,
    ) -> None:
        """Train the controller.

        Parameters
        ----------
        env:
            Function returning the profit for a given action.  The reward is
            computed as ``profit - penalty * complexity``.
        episodes:
            Number of training episodes.
        epsilon:
            Exploration rate for epsilon-greedy policy.
        alpha:
            Learning rate for incremental update of action values.
        penalty:
            Multiplicative penalty applied to model/feature complexity.
        """

        for _ in range(episodes):
            action = self.choose_action(epsilon)
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
            key = self._key(action)
            old = self.q_values.get(key, 0.0)
            self.q_values[key] = old + alpha * (reward - old)

        self._save()

    def select_best(self) -> Action | None:
        if not self.q_values:
            return None
        best_key = max(self.q_values, key=self.q_values.get)
        return self._decode_key(best_key)


def main() -> None:
    """Minimal example when executed as a script."""

    def toy_env(action: Action) -> float:
        subset, model = action
        # Toy profit: number of features + random noise minus model weight
        base = len(subset) * 0.5 + random.random()
        return base + (1.0 if model == "linear" else 1.5)

    controller = AutoMLController(
        features=["a", "b", "c"],
        models={"linear": 1, "tree": 2},
    )
    controller.train(toy_env, episodes=10)
    print("Best action:", controller.select_best())


if __name__ == "__main__":  # pragma: no cover - manual usage example
    main()
