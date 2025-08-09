#!/usr/bin/env python3
"""Train a simple RL agent from trade logs."""

import argparse
import json
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import logging
import pandas as pd

import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

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

try:  # pragma: no cover - optional dependency
    from self_play_env import SelfPlayEnv  # type: ignore
except Exception:  # pragma: no cover - fallback when executed from repo root
    try:
        from scripts.self_play_env import SelfPlayEnv  # type: ignore
    except Exception:  # pragma: no cover - simulation optional
        SelfPlayEnv = None  # type: ignore


# OpenTelemetry setup
resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "train_rl_agent")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
elif os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST") and JaegerExporter:
    provider.add_span_processor(
        BatchSpanProcessor(
            JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )
        )
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


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
    for log_file in sorted(data_dir.glob("trades*.csv")):
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
    invalid_rows = pd.DataFrame(columns=df_logs.columns)
    if "event_id" in df_logs.columns:
        dup_mask = df_logs.duplicated(subset="event_id", keep="first")
        if dup_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[dup_mask]])
            logging.warning("Dropping %s duplicate event_id rows", dup_mask.sum())
        df_logs = df_logs[~dup_mask]

    if set(["ticket", "action"]).issubset(df_logs.columns):
        crit_mask = (
            df_logs["ticket"].isna()
            | (df_logs["ticket"].astype(str) == "")
            | df_logs["action"].isna()
            | (df_logs["action"].astype(str) == "")
        )
        if crit_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[crit_mask]])
            logging.warning("Dropping %s rows with missing ticket/action", crit_mask.sum())
        df_logs = df_logs[~crit_mask]

    if "lots" in df_logs.columns:
        df_logs["lots"] = pd.to_numeric(df_logs["lots"], errors="coerce")
    if "price" in df_logs.columns:
        df_logs["price"] = pd.to_numeric(df_logs["price"], errors="coerce")
    unreal_mask = pd.Series(False, index=df_logs.index)
    if "lots" in df_logs.columns:
        unreal_mask |= df_logs["lots"] < 0
    if "price" in df_logs.columns:
        unreal_mask |= df_logs["price"].isna()
    if unreal_mask.any():
        invalid_rows = pd.concat([invalid_rows, df_logs[unreal_mask]])
        logging.warning("Dropping %s rows with negative lots or NaN price", unreal_mask.sum())
    df_logs = df_logs[~unreal_mask]

    if not invalid_rows.empty:
        invalid_file = data_dir / "invalid_rows.csv"
        try:
            invalid_rows.to_csv(invalid_file, index=False)
        except Exception:  # pragma: no cover - disk issues
            pass

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
# Dataset helpers
# -------------------------------

def _build_dataset(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DictVectorizer]:
    """Return (states, actions, rewards, next_states, vectorizer).

    The dataset is constructed purely from the trade logs without any
    environment interaction.  Each trade becomes a state-action-reward
    tuple and the next state is simply the next trade's state.
    """

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
    states = vec.fit_transform(feats).astype(np.float32)
    actions_arr = np.asarray(actions, dtype=int)
    rewards_arr = np.asarray(rewards, dtype=float)
    if len(states) > 1:
        next_states = np.vstack([states[1:], states[-1:]])
    else:  # pragma: no cover - single trade edge case
        next_states = states.copy()

    return states, actions_arr, rewards_arr, next_states, vec


# -------------------------------
# RL Training
# -------------------------------

def train(
    data_dir: Path,
    out_dir: Path,
    *,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    training_steps: int = 10,
    batch_size: int = 4,
    buffer_size: int = 100,
    update_freq: int = 1,
    algo: str = "dqn",
    start_model: Path | None = None,
    compress_model: bool = False,
    self_play: bool = False,
) -> None:
    """Train a small RL agent from ``data_dir``."""
    states, actions, rewards, next_states, vec = _build_dataset(data_dir)
    n_features = states.shape[1]

    init_model_data = None
    if start_model is not None and start_model.exists():
        try:
            with open(start_model) as f:
                init_model_data = json.load(f)
        except Exception:
            init_model_data = None

    algo_key = algo.lower()

    if self_play and algo_key not in {"ppo", "dqn"}:
        raise ValueError("--self-play requires ppo or dqn algorithm")

    # precompute experience tuples for offline algorithms
    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = [
        (states[i], int(actions[i]), float(rewards[i]), next_states[i])
        for i in range(len(actions))
    ]

    if algo_key in {"ppo", "dqn"}:
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is not installed")
        if self_play and SelfPlayEnv is None:
            raise ImportError("SelfPlayEnv is required for self-play")

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
        model_cls = algo_map[algo_key]
        model = model_cls("MlpPolicy", env, verbose=0)
        if self_play:
            sim_env = SelfPlayEnv()
            half = max(1, training_steps // 2)
            model.learn(total_timesteps=half)
            model.set_env(sim_env)
            model.learn(total_timesteps=training_steps - half)
        else:
            model.learn(total_timesteps=training_steps)

        if self_play:
            model.set_env(env)
        preds: List[int] = []
        total_r = 0.0
        obs, _ = env.reset()
        for _ in range(len(actions)):
            act, _ = model.predict(obs, deterministic=True)
            preds.append(int(act))
            obs, reward, done, _, _ = env.step(int(act))
            total_r += float(reward)
            if done:
                break
        train_acc = float(np.mean(np.array(preds) == np.array(actions)))
        sim_metrics = None
        if self_play:
            sim_obs, _ = sim_env.reset()
            sim_total = 0.0
            for _ in range(sim_env.steps):
                sim_act, _ = model.predict(sim_obs, deterministic=True)
                sim_obs, r, done, _, _ = sim_env.step(int(sim_act))
                sim_total += float(r)
                if done:
                    break
            sim_metrics = {
                "total_reward": float(sim_total),
                "avg_reward": float(sim_total / sim_env.steps),
            }
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / "model_weights"
        model.save(str(weights_path))
        model_info = {
            "model_id": "rl_agent_sb3",
            "algo": algo_key,
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": vec.get_feature_names_out().tolist(),
            "train_accuracy": train_acc,
            "avg_reward": float(total_r / max(1, len(actions))),
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "val_accuracy": float("nan"),
            "accuracy": float("nan"),
            "num_samples": len(actions),
            "weights_file": weights_path.with_suffix(".zip").name,
        }
        if self_play:
            model_info["training_type"] = "self_play"
        elif init_model_data is not None:
            model_info["training_type"] = "supervised+rl"
            model_info["init_model"] = start_model.name if start_model else None
            model_info["init_model_id"] = init_model_data.get("model_id")
        else:
            model_info["training_type"] = "rl_only"
        if self_play:
            model_info["self_play_params"] = {
                "steps": sim_env.steps,
                "drift": sim_env.drift,
                "volatility": sim_env.volatility,
                "spread": sim_env.spread,
            }
            model_info["self_play_metrics"] = sim_metrics

        model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
        open_func = gzip.open if compress_model else open
        with open_func(model_path, "wt") as f:
            json.dump(model_info, f, indent=2)

        print(f"Model written to {model_path}")
        print(f"Weights written to {weights_path.with_suffix('.zip')}")
        return

    if algo_key not in {"qlearn", "cql"}:
        raise ValueError(f"Unsupported algorithm: {algo}")

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

    if algo_key == "cql":
        alpha = 0.01  # conservative penalty strength
        for _ in range(training_steps):
            for s, a, r, ns in experiences:
                q_next0 = intercepts[0] + np.dot(weights[0], ns)
                q_next1 = intercepts[1] + np.dot(weights[1], ns)
                q_target = r + gamma * max(q_next0, q_next1)
                q_current = intercepts[a] + np.dot(weights[a], s)
                td_err = q_target - q_current
                weights[a] += learning_rate * td_err * s
                intercepts[a] += learning_rate * td_err
                # conservative update for all actions
                for act in (0, 1):
                    q_val = intercepts[act] + np.dot(weights[act], s)
                    weights[act] -= learning_rate * alpha * q_val * s
                    intercepts[act] -= learning_rate * alpha * q_val
        training_type = "offline_rl"
        episode_rewards = [float(np.mean(rewards))]
        episode_totals = [float(np.sum(rewards))]
    else:  # qlearn
        episode_rewards: List[float] = []  # average reward per step
        episode_totals: List[float] = []   # total reward per episode
        for _ in range(training_steps):
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
        training_type = "rl_only" if init_model_data is None else "supervised+rl"
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
        "algo": algo_key,
        "feature_names": vec.get_feature_names_out().tolist(),
        "coefficients": (weights[0] - weights[1]).astype(np.float32).tolist(),
        "intercept": float(intercepts[0] - intercepts[1]),
        "q_weights": weights.astype(np.float32).tolist(),
        "q_intercepts": intercepts.astype(np.float32).tolist(),
        "train_accuracy": train_acc,
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_reward_per_episode": float(np.mean(episode_totals)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "training_steps": training_steps * len(experiences),
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "val_accuracy": float("nan"),
        "accuracy": float("nan"),
        "num_samples": len(actions),
    }
    model["training_type"] = training_type
    if training_type != "offline_rl" and init_model_data is not None:
        model["init_model"] = start_model.name if start_model is not None else None
        model["init_model_id"] = init_model_data.get("model_id")

    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {model_path}")


def main() -> None:
    span = tracer.start_span("train_rl_agent")
    ctx = span.get_span_context()
    print(
        f"trace_id={format_trace_id(ctx.trace_id)} span_id={format_span_id(ctx.span_id)}"
    )
    p = argparse.ArgumentParser(description="Train RL agent from logs")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--learning-rate", type=float, default=0.1, help="learning rate")
    p.add_argument("--epsilon", type=float, default=0.1, help="epsilon for exploration")
    p.add_argument("--training-steps", type=int, default=100, help="total training steps")
    p.add_argument("--batch-size", type=int, default=4, help="batch size for updates")
    p.add_argument("--buffer-size", type=int, default=100, help="replay buffer size")
    p.add_argument("--update-freq", type=int, default=1, help="steps between updates")
    p.add_argument(
        "--algo",
        default="dqn",
        help=(
            "RL algorithm: dqn (default) or ppo if stable-baselines3 is installed."
            " Pass qlearn for a simple numpy implementation or cql for offline"
            " conservative Q-learning."
        ),
    )
    p.add_argument("--start-model", help="path to initial model coefficients")
    p.add_argument("--compress-model", action="store_true", help="write model.json.gz")
    p.add_argument(
        "--self-play",
        action="store_true",
        help="alternate training on real logs and simulated data",
    )
    args = p.parse_args()
    train(
        Path(args.data_dir),
        Path(args.out_dir),
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_freq=args.update_freq,
        algo=args.algo,
        start_model=Path(args.start_model) if args.start_model else None,
        compress_model=args.compress_model,
        self_play=args.self_play,
    )
    span.end()


if __name__ == "__main__":
    main()
