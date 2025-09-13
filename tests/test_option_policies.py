import json
from pathlib import Path

import pytest

from tests import HAS_SB3, HAS_NUMPY

if HAS_NUMPY:
    import numpy as np

if HAS_SB3 and HAS_NUMPY:
    from scripts.train_rl_agent import train_options
    from scripts.replay_decisions import replay_option_policy
    from botcopier.rl.options import OptionTradeEnv, default_skills, evaluate_option_policy
    from stable_baselines3 import PPO

pytestmark = pytest.mark.skipif(
    not (HAS_SB3 and HAS_NUMPY), reason="stable-baselines3 or numpy not installed"
)


def _build_synthetic_dataset():
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


def test_option_policy_improves_reward(tmp_path: Path) -> None:
    states, actions, rewards = _build_synthetic_dataset()
    model_info = train_options(
        states,
        actions,
        rewards,
        tmp_path,
        training_steps=200,
        learning_rate=0.1,
    )
    weights_file = tmp_path / model_info["option_weights_file"]
    assert weights_file.exists()

    model = PPO.load(str(weights_file))
    env = OptionTradeEnv(states, actions, rewards, default_skills())
    trained_reward = evaluate_option_policy(model, env)

    env_rand = OptionTradeEnv(states, actions, rewards, default_skills())
    obs, _ = env_rand.reset()
    random_total = 0.0
    for _ in range(len(actions)):
        act = env_rand.action_space.sample()
        obs, r, done, _, _ = env_rand.step(int(act))
        random_total += r
        if done:
            break

    assert trained_reward > random_total

    # ensure replay script can load and evaluate
    stats = replay_option_policy(states, actions, rewards, model_info, tmp_path)
    assert stats["total_reward"] == pytest.approx(trained_reward)
