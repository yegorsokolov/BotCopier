#!/usr/bin/env python3
"""Train a simple RL agent from trade logs."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import stable_baselines3 as sb3  # type: ignore
    try:
        from gym import Env, spaces  # type: ignore
    except Exception:  # pragma: no cover - gymnasium fallback
        from gymnasium import Env, spaces  # type: ignore
    HAS_SB3 = True
except Exception:  # pragma: no cover - optional dependency
    sb3 = None  # type: ignore
    spaces = None  # type: ignore
    Env = object  # type: ignore
    HAS_SB3 = False

import numpy as np
from sklearn.feature_extraction import DictVectorizer


# -------------------------------
# Data loading utilities
# -------------------------------

def _load_logs(data_dir: Path) -> pd.DataFrame:
    """Load raw log rows from ``data_dir``."""

    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "comment",
        "remaining_lots",
    ]

    dfs: List[pd.DataFrame] = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        df = pd.read_csv(
            log_file,
            sep=";",
            names=fields,
            header=0,
            parse_dates=["event_time"],
        )
        dfs.append(df)

    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=fields)

    df_logs.columns = [c.lower() for c in df_logs.columns]

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    df_logs["action"] = df_logs["action"].fillna("").str.upper()
    df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    return df_logs


def _pair_trades(rows: List[Dict]) -> List[Dict]:
    """Pair OPEN and CLOSE rows into trade records."""
    open_map: Dict[str, Dict] = {}
    trades: List[Dict] = []
    for r in rows:
        action = (r.get("action") or "").upper()
        ticket = r.get("ticket")
        if action == "OPEN":
            open_map[ticket] = r
        elif action == "CLOSE" and ticket in open_map:
            o = open_map.pop(ticket)
            profit = float(r.get("profit", 0) or 0)
            trades.append({"open": o, "profit": profit})
    return trades


def _extract_feature(row: Dict) -> Dict:
    """Extract feature dictionary from an OPEN row."""
    t = row["event_time"]
    if not isinstance(t, datetime):
        parsed = None
        for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
            try:
                parsed = datetime.strptime(str(t), fmt)
                break
            except Exception:
                continue
        if parsed is None:
            parsed = datetime.utcnow()
        t = parsed

    price = float(row.get("price", 0) or 0)
    sl = float(row.get("sl", 0) or 0)
    tp = float(row.get("tp", 0) or 0)
    lots = float(row.get("lots", 0) or 0)

    return {
        "symbol": row.get("symbol", ""),
        "hour": t.hour,
        "lots": lots,
        "sl_dist": sl - price,
        "tp_dist": tp - price,
    }


# -------------------------------
# RL Training
# -------------------------------

def train(
    data_dir: Path,
    out_dir: Path,
    *,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 10,
    batch_size: int = 4,
    buffer_size: int = 100,
    update_freq: int = 1,
    algo: str = "qlearn",
    start_model: Path | None = None,
) -> None:
    """Train a small RL agent from ``data_dir``."""

    rows_df = _load_logs(data_dir)
    trades = _pair_trades(rows_df.to_dict("records"))

    if not trades:
        raise ValueError(f"No training data found in {data_dir}")

    feats: List[Dict] = []
    actions: List[int] = []
    rewards: List[float] = []
    for t in trades:
        o = t["open"]
        feats.append(_extract_feature(o))
        actions.append(0 if int(float(o.get("order_type", 0))) == 0 else 1)
        rewards.append(float(t["profit"]))

    vec = DictVectorizer(sparse=False)
    states = vec.fit_transform(feats)
    n_features = states.shape[1]

    init_model_data = None
    if start_model is not None and start_model.exists():
        try:
            with open(start_model) as f:
                init_model_data = json.load(f)
        except Exception:
            init_model_data = None

    if algo != "qlearn":
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is not installed")

        class TradeEnv(Env):
            def __init__(self, observations, rewards):
                super().__init__()
                self.observations = observations.astype(np.float32)
                self.rewards = np.asarray(rewards, dtype=np.float32)
                self.action_space = spaces.Discrete(2)
                self.observation_space = spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.observations.shape[1],),
                    dtype=np.float32,
                )
                self.idx = 0

            def reset(self, *, seed=None, options=None):  # type: ignore[override]
                self.idx = 0
                return self.observations[self.idx], {}

            def step(self, action):
                reward = float(self.rewards[self.idx])
                self.idx += 1
                done = self.idx >= len(self.observations)
                obs = (
                    self.observations[self.idx]
                    if not done
                    else self.observations[-1]
                )
                return obs, reward, done, False, {}

        env = TradeEnv(states, rewards)
        algo_map = {"ppo": sb3.PPO, "dqn": sb3.DQN}
        algo_key = algo.lower()
        if algo_key not in algo_map:
            raise ValueError(f"Unsupported algorithm: {algo}")
        model_cls = algo_map[algo_key]
        model = model_cls("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=episodes * len(actions))

        preds = []
        obs, _ = env.reset()
        for _ in range(len(actions)):
            act, _ = model.predict(obs, deterministic=True)
            preds.append(int(act))
            obs, _, done, _, _ = env.step(int(act))
            if done:
                break
        train_acc = float(np.mean(np.array(preds) == np.array(actions)))
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / "model_weights"
        model.save(str(weights_path))
        episode_total = float(np.sum(rewards))
        model_info = {
            "model_id": "rl_agent_sb3",
            "algo": algo_key,
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": vec.get_feature_names_out().tolist(),
            "train_accuracy": train_acc,
            "avg_reward": float(np.mean(rewards)),
            "avg_reward_per_episode": episode_total,
            "episode_rewards": [episode_total for _ in range(episodes)],
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "val_accuracy": float("nan"),
            "accuracy": float("nan"),
            "num_samples": len(actions),
            "weights_file": weights_path.with_suffix(".zip").name,
        }

        with open(out_dir / "model.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"Model written to {out_dir / 'model.json'}")
        print(f"Weights written to {weights_path.with_suffix('.zip')}")
        return

    # prepare experience tuples (state, action, reward, next_state)
    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
    for i in range(len(actions)):
        s = states[i]
        ns = states[i + 1] if i + 1 < len(actions) else states[i]
        experiences.append((s, actions[i], rewards[i], ns))

    weights = np.zeros((2, n_features))
    intercepts = np.zeros(2)
    if init_model_data is not None:
        feats_match = init_model_data.get("feature_names") == vec.get_feature_names_out().tolist()
        if feats_match:
            if "weights" in init_model_data and "intercepts" in init_model_data:
                try:
                    weights = np.array(init_model_data["weights"], dtype=float)
                    intercepts = np.array(init_model_data["intercepts"], dtype=float)
                except Exception:
                    weights = np.zeros((2, n_features))
                    intercepts = np.zeros(2)
            elif "coefficients" in init_model_data and "intercept" in init_model_data:
                coefs = np.array(init_model_data["coefficients"], dtype=float)
                bias = float(init_model_data["intercept"])
                weights = np.vstack([coefs / 2.0, -coefs / 2.0])
                intercepts = np.array([bias / 2.0, -bias / 2.0], dtype=float)
    gamma = 0.9

    episode_rewards: List[float] = []  # average reward per step
    episode_totals: List[float] = []   # total reward per episode
    for _ in range(episodes):
        total_r = 0.0
        buffer: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        for step, exp in enumerate(experiences):
            s, a, r, ns = exp
            total_r += r
            if len(buffer) >= buffer_size:
                buffer.pop(0)
            # add with initial priority weight of 1.0
            buffer.append((s, a, r, ns, 1.0))

            if step % update_freq == 0 and len(buffer) >= batch_size:
                weights_arr = np.array([b[4] for b in buffer], dtype=float)
                prob = weights_arr / weights_arr.sum()
                batch_idx = np.random.choice(len(buffer), size=batch_size, p=prob)
                for idx in batch_idx:
                    bs, ba, br, bns, bw = buffer[idx]
                    q_next0 = intercepts[0] + np.dot(weights[0], bns)
                    q_next1 = intercepts[1] + np.dot(weights[1], bns)
                    q_target = br + gamma * max(q_next0, q_next1)
                    q_current = intercepts[ba] + np.dot(weights[ba], bs)
                    td_err = q_target - q_current
                    weights[ba] += learning_rate * td_err * bs
                    intercepts[ba] += learning_rate * td_err
                    # update priority weight based on TD error
                    buffer[idx] = (bs, ba, br, bns, float(abs(td_err)) + 1e-6)
        episode_rewards.append(total_r / len(experiences))
        episode_totals.append(total_r)

    preds: List[int] = []
    for s in states:
        if np.random.rand() < epsilon:
            preds.append(np.random.randint(0, 2))
            continue
        qb = intercepts[0] + np.dot(weights[0], s)
        qs = intercepts[1] + np.dot(weights[1], s)
        preds.append(0 if qb >= qs else 1)
    train_acc = float(np.mean(np.array(preds) == np.array(actions)))

    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "model_id": "rl_agent",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "coefficients": (weights[0] - weights[1]).tolist(),
        "intercept": float(intercepts[0] - intercepts[1]),
        "q_weights": weights.tolist(),
        "q_intercepts": intercepts.tolist(),
        "train_accuracy": train_acc,
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_reward_per_episode": float(np.mean(episode_totals)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "val_accuracy": float("nan"),
        "accuracy": float("nan"),
        "num_samples": len(actions),
    }

    if init_model_data is not None:
        model["init_model"] = start_model.name if start_model is not None else None
        model["init_model_id"] = init_model_data.get("model_id")
        model["training_type"] = "supervised+rl"
    else:
        model["training_type"] = "rl_only"

    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {out_dir / 'model.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train RL agent from logs")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--learning-rate", type=float, default=0.1, help="learning rate")
    p.add_argument("--epsilon", type=float, default=0.1, help="epsilon for exploration")
    p.add_argument("--episodes", type=int, default=10, help="training episodes")
    p.add_argument("--batch-size", type=int, default=4, help="batch size for updates")
    p.add_argument("--buffer-size", type=int, default=100, help="replay buffer size")
    p.add_argument("--update-freq", type=int, default=1, help="steps between updates")
    p.add_argument(
        "--algo",
        default="qlearn",
        help="RL algorithm: qlearn (default), ppo or dqn if stable-baselines3 is installed",
    )
    p.add_argument("--start-model", help="path to initial model coefficients")
    args = p.parse_args()
    train(
        Path(args.data_dir),
        Path(args.out_dir),
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        episodes=args.episodes,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_freq=args.update_freq,
        algo=args.algo,
        start_model=Path(args.start_model) if args.start_model else None,
    )


if __name__ == "__main__":
    main()
