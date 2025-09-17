"""Convergence tests for the :mod:`automl.controller` module."""

from __future__ import annotations

import json
import random
from pathlib import Path

from automl.controller import Action, AutoMLController


def _toy_env(action: Action) -> float:
    rewards = {
        (("f1",), "m1"): 0.1,
        (("f1",), "m2"): 0.2,
        (("f2",), "m1"): 0.05,
        (("f2",), "m2"): 0.15,
        (("f1", "f2"), "m1"): 0.25,
        (("f1", "f2"), "m2"): 0.5,
    }
    return rewards[action]


def test_controller_converges(tmp_path: Path) -> None:
    random.seed(0)
    controller = AutoMLController(
        ["f1", "f2"],
        {"m1": 1, "m2": 2},
        model_path=tmp_path / "model.json",
        max_subset_size=2,
    )
    controller.train(_toy_env, episodes=500, alpha=0.2, penalty=0.01)
    best = controller.select_best()
    assert best == (("f1", "f2"), "m2")
    data = json.loads((tmp_path / "model.json").read_text())
    sec = data["automl_controller"]["best_action"]
    assert sec == {"features": ["f1", "f2"], "model": "m2"}
    last = data["automl_controller"].get("last_action")
    assert last == {"features": ["f1", "f2"], "model": "m2"}


def test_controller_reuse(tmp_path: Path) -> None:
    random.seed(0)
    path = tmp_path / "model.json"
    controller = AutoMLController(["f1", "f2"], {"m1": 1, "m2": 2}, model_path=path)
    controller.train(_toy_env, episodes=200, alpha=0.2, penalty=0.01)
    best = controller.select_best()
    reused = AutoMLController(["f1", "f2"], {"m1": 1, "m2": 2}, model_path=path, reuse=True)
    assert reused.select_best() == best
    assert reused.last_action in reused.action_space
    fresh = AutoMLController(["f1", "f2"], {"m1": 1, "m2": 2}, model_path=path, reuse=False)
    assert fresh.select_best() != best


def test_controller_limits_action_growth(tmp_path: Path) -> None:
    random.seed(0)
    features = [f"f{i}" for i in range(30)]
    controller = AutoMLController(
        features,
        {"m1": 1, "m2": 2},
        model_path=tmp_path / "model.json",
        max_subset_size=2,
        episode_sample_size=5,
        reuse=False,
    )

    def constant_env(action: Action) -> float:
        return float(len(action[0])) * 0.1

    controller.train(constant_env, episodes=10, alpha=0.1, penalty=0.0)
    actions = controller.action_space
    assert actions
    assert max(len(subset) for subset, _ in actions) <= 2
    assert len(actions) <= 10 * 5 * len(controller.models)
