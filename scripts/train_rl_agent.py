#!/usr/bin/env python3
"""Train a simple RL agent from trade logs."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer


# -------------------------------
# Data loading utilities
# -------------------------------

def _load_logs(data_dir: Path) -> List[Dict]:
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

    rows: List[Dict] = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        with open(log_file, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) == len(fields):
                    rows.append(dict(zip(fields, row)))
                else:
                    r = {fields[i]: row[i] for i in range(min(len(row), len(fields)))}
                    rows.append(r)
    return rows


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
    try:
        t = datetime.strptime(row["event_time"], "%Y.%m.%d %H:%M:%S")
    except ValueError:
        t = datetime.strptime(row["event_time"], "%Y.%m.%d %H:%M")

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
) -> None:
    """Train a very small Q-learning agent from ``data_dir``."""

    rows = _load_logs(data_dir)
    trades = _pair_trades(rows)

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

    # prepare experience tuples (state, action, reward, next_state)
    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
    for i in range(len(actions)):
        s = states[i]
        ns = states[i + 1] if i + 1 < len(actions) else states[i]
        experiences.append((s, actions[i], rewards[i], ns))

    weights = np.zeros((2, n_features))
    intercepts = np.zeros(2)
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
    )


if __name__ == "__main__":
    main()
