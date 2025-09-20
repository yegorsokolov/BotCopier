from __future__ import annotations

"""Policy-gradient controller for AutoML feature/model selection."""

import itertools
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - registry may not be present in tests
    from botcopier.models.registry import MODEL_REGISTRY as _MODEL_REGISTRY
except Exception:  # pragma: no cover - best effort
    _MODEL_REGISTRY = {}

Action = Tuple[Tuple[str, ...], str]

DEFAULT_EPISODE_COMBINATION_CAP = 4096

__all__ = ["AutoMLController", "Action"]


class AutoMLController:
    """Simple REINFORCE-style policy gradient controller.

    Parameters
    ----------
    features:
        Iterable of available feature names.
    models:
        Mapping of model name to an integer representing its relative
        complexity. Higher values imply greater complexity penalty.  If
        ``None`` all models currently registered in
        :mod:`botcopier.models.registry` are used with a default complexity of 1.
    model_path:
        File used to persist the controller's policy and best action.
    reuse:
        When ``True`` the controller attempts to load an existing policy from
        ``model_path``. When ``False`` any existing policy is ignored.
    """

    def __init__(
        self,
        features: Iterable[str],
        models: Dict[str, int] | None = None,
        model_path: str | os.PathLike[str] = "model.json",
        *,
        reuse: bool = True,
        max_subset_size: int | None = None,
        episode_sample_size: int | None = None,
        baseline_momentum: float | None = 0.9,
        episode_combination_cap: int | None = None,
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise ValueError("AutoMLController requires at least one feature")
        if models is None:
            models = {name: 1 for name in _MODEL_REGISTRY.keys()}
        if not models:
            raise ValueError("AutoMLController requires at least one model")
        self.models = models
        self.model_path = Path(model_path)
        self.max_subset_size = (
            len(self.features) if max_subset_size is None else max_subset_size
        )
        if self.max_subset_size < 1:
            raise ValueError("max_subset_size must be at least 1")
        self.max_subset_size = min(self.max_subset_size, len(self.features))
        self.episode_sample_size = episode_sample_size
        if self.episode_sample_size is not None and self.episode_sample_size < 1:
            raise ValueError("episode_sample_size must be positive when provided")

        if episode_combination_cap is None:
            episode_combination_cap = DEFAULT_EPISODE_COMBINATION_CAP
        if episode_combination_cap < 1:
            raise ValueError("episode_combination_cap must be at least 1")
        self.episode_combination_cap = episode_combination_cap

        self.baseline_momentum = baseline_momentum
        if self.baseline_momentum is not None:
            if self.baseline_momentum <= 0.0:
                self.baseline_momentum = None
            elif self.baseline_momentum > 1.0:
                raise ValueError("baseline_momentum must be between 0 and 1")

        self._known_actions: Dict[str, Action] = {}
        self._current_actions: List[Action] = []
        self._current_keys: List[str] = []
        self._episode_seen_keys: set[str] = set()
        self._episode_seen_subsets: set[Tuple[str, ...]] = set()
        self.telemetry: Dict[str, float | int | bool] = {
            "total_subsets": 0,
            "sampled_subsets": 0,
            "total_actions": 0,
            "enumerated_actions": 0,
            "explored_actions": 0,
            "explored_subsets": 0,
            "coverage": 0.0,
            "combination_cap": self.episode_combination_cap,
            "random_sampling": False,
            "remaining_actions": 0,
        }
        self.last_action: Action | None = None

        # policy parameters and reward estimates per action
        self.theta: Dict[str, float] = {}
        self.avg_reward: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

        # baseline critic statistics
        self._baseline: float = 0.0
        self._baseline_updates: int = 0

        if reuse:
            self._load()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the learned policy."""
        self.theta.clear()
        self.avg_reward.clear()
        self.counts.clear()
        self._known_actions.clear()
        self._current_actions = []
        self._current_keys = []
        self._episode_seen_keys.clear()
        self._episode_seen_subsets.clear()
        self.last_action = None
        self._baseline = 0.0
        self._baseline_updates = 0
        self.telemetry.update(
            {
                "total_subsets": 0,
                "sampled_subsets": 0,
                "total_actions": 0,
                "enumerated_actions": 0,
                "explored_actions": 0,
                "explored_subsets": 0,
                "coverage": 0.0,
                "random_sampling": False,
                "remaining_actions": 0,
            }
        )

    def _key(self, action: Action) -> str:
        subset, model = action
        subset_key = ",".join(sorted(subset))
        return f"{subset_key}|{model}"

    def _decode(self, key: str) -> Action:
        subset_key, model = key.split("|")
        subset = tuple(filter(None, subset_key.split(",")))
        return subset, model

    def _register_action(self, action: Action) -> str:
        key = self._key(action)
        if key not in self._known_actions:
            self._known_actions[key] = action
        self.theta.setdefault(key, 0.0)
        self.avg_reward.setdefault(key, 0.0)
        self.counts.setdefault(key, 0)
        return key

    def configure(
        self,
        *,
        max_subset_size: int | None = None,
        episode_sample_size: int | None = None,
        baseline_momentum: float | None = None,
        episode_combination_cap: int | None = None,
    ) -> None:
        """Update controller hyperparameters for future episodes."""

        if max_subset_size is not None:
            if max_subset_size < 1:
                raise ValueError("max_subset_size must be at least 1")
            self.max_subset_size = min(max_subset_size, len(self.features))
        if episode_sample_size is not None:
            if episode_sample_size < 1:
                raise ValueError("episode_sample_size must be positive")
            self.episode_sample_size = episode_sample_size
        if episode_combination_cap is not None:
            if episode_combination_cap < 1:
                raise ValueError("episode_combination_cap must be at least 1")
            self.episode_combination_cap = episode_combination_cap
            self.telemetry["combination_cap"] = self.episode_combination_cap
        if baseline_momentum is not None:
            if baseline_momentum <= 0.0:
                self.baseline_momentum = None
            elif baseline_momentum > 1.0:
                raise ValueError("baseline_momentum must be between 0 and 1")
            else:
                self.baseline_momentum = baseline_momentum
        self._current_actions = []
        self._current_keys = []
        self._episode_seen_keys.clear()
        self._episode_seen_subsets.clear()

    def _iter_subsets(self) -> Iterable[Tuple[str, ...]]:
        max_r = min(self.max_subset_size, len(self.features))
        for r in range(1, max_r + 1):
            yield from itertools.combinations(self.features, r)

    def _random_subset(self) -> Tuple[str, ...]:
        max_r = min(self.max_subset_size, len(self.features))
        if max_r <= 0:
            raise ValueError("AutoMLController requires at least one feature")
        size = random.randint(1, max_r)
        if size == 0:
            return ()
        return tuple(sorted(random.sample(self.features, size)))

    def _prepare_episode_actions(self, include: Sequence[Action] | None = None) -> None:
        include = list(include or [])
        subset_map: Dict[Tuple[str, ...], None] = OrderedDict()
        include_actions: List[Action] = []
        for action in include:
            subset = tuple(sorted(action[0]))
            model = action[1]
            include_action = (subset, model)
            include_actions.append(include_action)
            subset_map.setdefault(subset, None)

        feature_count = len(self.features)
        max_r = min(self.max_subset_size, feature_count)
        total_subsets = sum(math.comb(feature_count, r) for r in range(1, max_r + 1))
        total_actions = total_subsets * len(self.models)

        subset_target = len(subset_map)
        random_sampling = False
        subset_cap = max(1, math.ceil(self.episode_combination_cap / len(self.models)))

        if self.episode_sample_size is not None:
            random_sampling = True
            subset_target = max(subset_target, min(self.episode_sample_size, total_subsets))
            if total_actions > self.episode_combination_cap:
                subset_target = min(subset_target, subset_cap)
        elif total_actions > self.episode_combination_cap:
            random_sampling = True
            subset_target = max(subset_target, min(total_subsets, subset_cap))
        else:
            subset_target = max(subset_target, total_subsets)

        if random_sampling:
            attempts = 0
            max_attempts = max(100, subset_target * 10)
            while len(subset_map) < subset_target and attempts < max_attempts:
                subset_map.setdefault(self._random_subset(), None)
                attempts += 1
            if len(subset_map) < subset_target:
                for subset in self._iter_subsets():
                    subset_map.setdefault(tuple(subset), None)
                    if len(subset_map) >= subset_target:
                        break
        else:
            for subset in self._iter_subsets():
                subset_map.setdefault(tuple(subset), None)
                if len(subset_map) >= subset_target:
                    break

        actions: List[Action] = []
        seen_keys: set[str] = set()

        for action in include_actions:
            key = self._register_action(action)
            if key in seen_keys:
                continue
            actions.append(action)
            seen_keys.add(key)
            if len(actions) >= self.episode_combination_cap:
                break

        subsets = list(subset_map.keys())
        random.shuffle(subsets)
        for subset in subsets:
            for model in self.models:
                action = (subset, model)
                key = self._register_action(action)
                if key in seen_keys:
                    continue
                actions.append(action)
                seen_keys.add(key)
                if len(actions) >= self.episode_combination_cap:
                    break
            if len(actions) >= self.episode_combination_cap:
                break

        if not actions:
            raise ValueError(
                "AutoMLController requires at least one feature/model combination"
            )

        random.shuffle(actions)
        self._current_actions = actions
        self._current_keys = [self._key(action) for action in actions]
        self._episode_seen_keys.clear()
        self._episode_seen_subsets.clear()
        enumerated_actions = len(actions)
        self.telemetry.update(
            {
                "total_subsets": total_subsets,
                "sampled_subsets": len(subset_map),
                "total_actions": total_actions,
                "enumerated_actions": enumerated_actions,
                "explored_actions": 0,
                "explored_subsets": 0,
                "coverage": 0.0,
                "combination_cap": self.episode_combination_cap,
                "random_sampling": random_sampling or total_actions > self.episode_combination_cap,
                "remaining_actions": enumerated_actions,
            }
        )

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
        last_action = sec.get("last_action")
        baseline = sec.get("baseline")
        baseline_updates = sec.get("baseline_updates")

        if theta:
            for k, v in theta.items():
                action = self._decode(k)
                self._register_action(action)
                self.theta[k] = float(v)
        if rewards:
            for k, v in rewards.items():
                action = self._decode(k)
                self._register_action(action)
                self.avg_reward[k] = float(v)
        if counts:
            for k, v in counts.items():
                action = self._decode(k)
                self._register_action(action)
                self.counts[k] = int(v)
        if isinstance(baseline, (int, float)):
            self._baseline = float(baseline)
        if isinstance(baseline_updates, int):
            self._baseline_updates = max(0, baseline_updates)
        if last_action and isinstance(last_action, dict):
            subset = tuple(last_action.get("features", ()))
            model = last_action.get("model")
            if subset and isinstance(model, str):
                key = self._key((subset, model))
                if key in self.theta:
                    self.last_action = (subset, model)

    def _save(self) -> None:
        try:
            data = json.loads(self.model_path.read_text())
        except Exception:
            data = {}
        sec = data.setdefault("automl_controller", {})
        sec["policy"] = self.theta
        sec["avg_reward"] = self.avg_reward
        sec["counts"] = self.counts
        if self.baseline_momentum is not None:
            sec["baseline"] = self._baseline
            sec["baseline_updates"] = self._baseline_updates
        if self.last_action is not None:
            sec["last_action"] = {
                "features": list(self.last_action[0]),
                "model": self.last_action[1],
            }
        else:
            sec.pop("last_action", None)
        best = self.select_best()
        if best is not None:
            sec["best_action"] = {"features": list(best[0]), "model": best[1]}
        with self.model_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Policy logic
    # ------------------------------------------------------------------
    def _probs(self, keys: Sequence[str]) -> List[float]:
        weights = [self.theta.get(k, 0.0) for k in keys]
        max_w = max(weights)
        exps = [math.exp(w - max_w) for w in weights]
        s = sum(exps)
        if s == 0.0:
            return [1.0 / len(keys)] * len(keys)
        return [e / s for e in exps]

    def sample_action(self) -> Tuple[Action, List[float]]:
        """Sample an action according to the current policy."""
        if not self._current_actions:
            self._prepare_episode_actions()

        available = [
            (action, key)
            for action, key in zip(self._current_actions, self._current_keys)
            if key not in self._episode_seen_keys
        ]
        if not available:
            self._prepare_episode_actions()
            available = [
                (action, key)
                for action, key in zip(self._current_actions, self._current_keys)
                if key not in self._episode_seen_keys
            ]
            if not available:
                raise RuntimeError("AutoMLController has no unseen actions to sample")

        self._current_actions = [action for action, _ in available]
        self._current_keys = [key for _, key in available]
        probs = self._probs(self._current_keys)
        index = random.choices(range(len(self._current_actions)), probs)[0]
        action = self._current_actions[index]
        key = self._current_keys[index]
        self._episode_seen_keys.add(key)
        subset = tuple(sorted(action[0]))
        self._episode_seen_subsets.add(subset)

        explored_actions = len(self._episode_seen_keys)
        enumerated_actions = int(self.telemetry.get("enumerated_actions", 0))
        self.telemetry["explored_actions"] = explored_actions
        self.telemetry["explored_subsets"] = len(self._episode_seen_subsets)
        if enumerated_actions > 0:
            self.telemetry["coverage"] = min(1.0, explored_actions / enumerated_actions)
            self.telemetry["remaining_actions"] = max(0, enumerated_actions - explored_actions)
        else:
            self.telemetry["coverage"] = 0.0
            self.telemetry["remaining_actions"] = 0

        self.last_action = action
        return action, probs

    def update(self, action: Action, reward: float, alpha: float = 0.1) -> None:
        """Update policy parameters for ``action`` with ``reward``."""
        key = self._register_action(action)
        if not self._current_actions or key not in self._current_keys:
            self._prepare_episode_actions(include=[action])

        probs = self._probs(self._current_keys)
        chosen = self._current_keys.index(key)

        advantage = reward
        if self.baseline_momentum is not None:
            advantage = reward - self._baseline
            if self._baseline_updates == 0:
                self._baseline = reward
            else:
                momentum = self.baseline_momentum
                if momentum is not None:
                    self._baseline = momentum * self._baseline + (1 - momentum) * reward
            self._baseline_updates += 1

        for i, k in enumerate(self._current_keys):
            grad = (1.0 if i == chosen else 0.0) - probs[i]
            self.theta[k] += alpha * advantage * grad
        self.counts[key] += 1
        c = self.counts[key]
        self.avg_reward[key] += (reward - self.avg_reward[key]) / c
        self.last_action = action
        self._current_actions = []
        self._current_keys = []
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
            self._prepare_episode_actions()
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

    @property
    def action_space(self) -> List[Action]:  # pragma: no cover - compatibility shim
        """Return the set of actions discovered so far."""

        return list(self._known_actions.values())
