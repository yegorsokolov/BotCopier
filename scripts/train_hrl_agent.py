#!/usr/bin/env python3
"""Train a hierarchical reinforcement learning agent.

This script demonstrates training a simple two level hierarchy where a
meta-controller selects a trading regime and sub-policies decide on
entry/exit actions.  It relies on :mod:`stable_baselines3` when available,
falling back to no-op stubs otherwise so the script remains importable on
systems without the optional dependency.

The resulting hierarchy is saved into ``model.json`` with a ``hierarchy``
field that contains metadata and (when SB3 is available) learned
parameters.  ``generate_mql4_from_model.py`` can embed this metadata in
rendered MQL4 strategies.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

try:  # pragma: no cover - optional dependency
    import stable_baselines3 as sb3  # type: ignore
    from gym import Env, spaces  # type: ignore
    HAS_SB3 = True
except Exception:  # pragma: no cover - SB3 not installed
    sb3 = None  # type: ignore
    Env = object  # type: ignore
    spaces = None  # type: ignore
    HAS_SB3 = False

try:  # pragma: no cover - used when running from repository root
    from self_play_env import SelfPlayEnv  # type: ignore
except Exception:  # pragma: no cover - import fallback
    try:
        from scripts.self_play_env import SelfPlayEnv  # type: ignore
    except Exception:  # pragma: no cover - environment optional
        SelfPlayEnv = None  # type: ignore


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------

class RegimeSelectionEnv(Env):
    """Meta-controller environment.

    The meta-controller chooses among ``n_regimes`` options.  Each step
    simply forwards to the underlying trading environment using a random
    action so that a learning algorithm receives a reward signal.  This is
    intentionally simple – the goal is to provide a minimal example that can
    be expanded for real training pipelines.
    """

    def __init__(self, base_env: Env, n_regimes: int) -> None:
        self.base_env = base_env
        self.n_regimes = n_regimes
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Discrete(n_regimes)

    def reset(self) -> Any:
        return self.base_env.reset()

    def step(self, action: int):
        # Select a random action in the underlying environment to generate a
        # reward signal for the chosen regime.
        sub_action = self.base_env.action_space.sample()
        obs, reward, done, info = self.base_env.step(sub_action)
        info = dict(info)
        info["regime"] = int(action)
        return obs, reward, done, info


class RegimeEnv(Env):
    """Wrapper used to train a sub-policy for a specific regime."""

    def __init__(self, base_env: Env, regime: int) -> None:
        self.base_env = base_env
        self.regime = regime
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(self) -> Any:  # pragma: no cover - simple forwarding
        return self.base_env.reset()

    def step(self, action):  # pragma: no cover - simple forwarding
        return self.base_env.step(action)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _to_jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SB3 parameters (numpy arrays) to plain lists for JSON."""

    simple: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            simple[k] = v.tolist()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - non-array values
            simple[k] = v
    return simple


def train(output: Path, meta_steps: int = 1000, sub_steps: int = 1000, regimes: int = 2) -> None:
    """Train the hierarchical agent and write ``model.json``."""

    if not SelfPlayEnv:
        raise RuntimeError("SelfPlayEnv is not available")

    env = SelfPlayEnv()

    hierarchy: Dict[str, Any] = {"type": "options", "sub_policies": {}}

    if HAS_SB3:
        # Train meta-controller
        meta_env = RegimeSelectionEnv(env, regimes)
        meta_model = sb3.PPO("MlpPolicy", meta_env, verbose=0)
        meta_model.learn(total_timesteps=meta_steps)
        hierarchy["meta_controller"] = {
            "algorithm": "PPO",
            "parameters": _to_jsonable(meta_model.get_parameters()),
        }

        # Train sub-policies for each regime
        for r in range(regimes):
            sub_env = RegimeEnv(env, r)
            sub_model = sb3.PPO("MlpPolicy", sub_env, verbose=0)
            sub_model.learn(total_timesteps=sub_steps)
            hierarchy["sub_policies"][str(r)] = {
                "algorithm": "PPO",
                "parameters": _to_jsonable(sub_model.get_parameters()),
            }
    else:  # pragma: no cover - executed when SB3 missing
        logging.warning("stable-baselines3 not installed; producing empty hierarchy")
        hierarchy["meta_controller"] = {}

    data = {"hierarchy": hierarchy}
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Model with hierarchy saved to {output}")


def main() -> None:  # pragma: no cover - thin CLI wrapper
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("model.json"))
    p.add_argument("--meta-steps", type=int, default=1000)
    p.add_argument("--sub-steps", type=int, default=1000)
    p.add_argument("--regimes", type=int, default=2)
    args = p.parse_args()
    train(args.output, args.meta_steps, args.sub_steps, args.regimes)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
