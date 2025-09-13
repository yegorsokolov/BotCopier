#!/usr/bin/env python3
"""Train a simple RL agent from trade logs."""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .model_fitting import load_logs
except ImportError:
    from model_fitting import load_logs

import logging
import os

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pyarrow.flight as flight
except Exception:  # pragma: no cover - optional dependency
    flight = None  # type: ignore
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

try:
    import stable_baselines3 as sb3  # type: ignore

    try:
        from stable_baselines3.common.buffers import (
            PrioritizedReplayBuffer,  # type: ignore
        )

        HAS_PRB = True
    except Exception:  # pragma: no cover - optional dependency
        PrioritizedReplayBuffer = None  # type: ignore
        HAS_PRB = False
    try:
        from gym import Env, spaces  # type: ignore
    except Exception:  # pragma: no cover - gymnasium fallback
        from gymnasium import Env, spaces  # type: ignore
    from stable_baselines3.common.env_util import make_vec_env  # type: ignore
    from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore

    HAS_SB3 = True
except Exception:  # pragma: no cover - optional dependency
    sb3 = None  # type: ignore
    spaces = None  # type: ignore
    Env = object  # type: ignore
    make_vec_env = None  # type: ignore
    SubprocVecEnv = None  # type: ignore
    HAS_SB3 = False

try:  # pragma: no cover - optional dependency
    import sb3_contrib as sb3c  # type: ignore

    HAS_SB3_CONTRIB = True
except Exception:  # pragma: no cover - optional dependency
    sb3c = None  # type: ignore
    HAS_SB3_CONTRIB = False

try:  # pragma: no cover - optional dependency
    from federated_buffer import FederatedBufferClient  # type: ignore
except Exception:  # pragma: no cover - executed when run from repo root
    try:
        from scripts.federated_buffer import FederatedBufferClient  # type: ignore
    except Exception:  # pragma: no cover - federated buffer optional
        FederatedBufferClient = None  # type: ignore

import numpy as np

from botcopier.rl.options import OptionTradeEnv, default_skills, evaluate_option_policy
from botcopier.utils.random import set_seed


def _max_drawdown(returns: np.ndarray) -> float:
    """Return the maximum drawdown of ``returns``."""
    if returns.size == 0:
        return 0.0
    cum = np.cumsum(returns, dtype=float)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd))


def _var_95(returns: np.ndarray) -> float:
    """Return the 95% Value at Risk of ``returns``."""
    if returns.size == 0:
        return 0.0
    return float(-np.quantile(returns, 0.05))

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction import DictVectorizer
except Exception:  # pragma: no cover - minimal fallback

    class DictVectorizer:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, feats):
            keys = sorted({k for f in feats for k in f.keys()})
            self.feature_names_ = keys
            rows = []
            for f in feats:
                row = []
                for k in keys:
                    v = f.get(k, 0.0)
                    try:
                        row.append(float(v))
                    except Exception:
                        row.append(0.0)
                rows.append(row)
            return np.array(rows, dtype=np.float32)

        def get_feature_names_out(self):  # pragma: no cover - simple
            return np.array(self.feature_names_)


try:  # pragma: no cover - optional dependency
    from self_play_env import SelfPlayEnv, train_self_play  # type: ignore
except Exception:  # pragma: no cover - fallback when executed from repo root
    try:
        from scripts.self_play_env import SelfPlayEnv, train_self_play  # type: ignore
    except Exception:  # pragma: no cover - simulation optional
        SelfPlayEnv = None  # type: ignore
        train_self_play = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import DecisionTransformerConfig, DecisionTransformerModel

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    DecisionTransformerConfig = DecisionTransformerModel = None  # type: ignore
    HAS_TRANSFORMERS = False


# OpenTelemetry setup
resource = Resource.create(
    {"service.name": os.getenv("OTEL_SERVICE_NAME", "train_rl_agent")}
)
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

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
    )
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {"level": record.levelname}
        if isinstance(record.msg, dict):
            log.update(record.msg)
        else:
            log["message"] = record.getMessage()
        if hasattr(record, "trace_id"):
            log["trace_id"] = format_trace_id(record.trace_id)
        if hasattr(record, "span_id"):
            log["span_id"] = format_span_id(record.span_id)
        return json.dumps(log)


logger = logging.getLogger(__name__)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------------
# Metrics feedback helper
# -------------------------------


def _send_live_metrics(url: str, metrics: Dict) -> Dict:
    """POST metrics to ``url`` and return any hyperparameter updates."""
    if not requests:
        return {}
    try:
        resp = requests.post(url, json=metrics, timeout=5)
        if resp.ok:
            return resp.json()  # type: ignore[return-value]
    except Exception:
        pass
    return {}


def train_options(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    out_dir: Path,
    *,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    training_steps: int = 1000,
    algo: str = "ppo",
    compress_model: bool = False,
    feature_names: List[str] | None = None,
) -> Dict:
    """Train a high-level option policy selecting among predefined skills."""

    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for option training")

    skills = default_skills()
    env = OptionTradeEnv(states, actions, rewards, skills)
    algo_key = algo.lower().replace("-", "_")
    algo_map = {"ppo": sb3.PPO, "dqn": sb3.DQN, "a2c": sb3.A2C}
    model_cls = algo_map.get(algo_key, sb3.PPO)
    model = model_cls(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=0,
    )
    model.learn(total_timesteps=training_steps)

    total_reward = evaluate_option_policy(model, env)

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "option_policy"
    model.save(str(weights_path))

    model_info = {
        "model_id": "option_rl_agent",
        "algo": algo_key,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names or [],
        "options": ["entry", "exit", "risk"],
        "option_weights_file": weights_path.with_suffix(".zip").name,
        "total_reward": float(total_reward),
        "training_steps": training_steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
    }

    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model_info, f, indent=2)

    return model_info


# -------------------------------
# Data loading utilities
# -------------------------------


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
    flight_uri: str | None = None,
    kafka_brokers: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DictVectorizer]:
    """Return (states, actions, rewards, next_states, vectorizer).

    The dataset is constructed purely from the trade logs without any
    environment interaction.  Each trade becomes a state-action-reward
    tuple and the next state is simply the next trade's state.
    """

    rows_df, _, _ = load_logs(data_dir, flight_uri=flight_uri)
    if kafka_brokers:
        raise NotImplementedError("kafka_brokers not supported")
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
    gamma: float = 0.9,
    replay_alpha: float = 0.6,
    replay_beta: float = 0.4,
    n_step: int = 1,
    algo: str = "dqn",
    num_envs: int = 1,
    start_model: Path | None = None,
    compress_model: bool = False,
    self_play: bool = False,
    flight_uri: str | None = None,
    kafka_brokers: str | None = None,
    federated_server: str | None = None,
    sync_interval: int = 50,
    metrics_url: str | None = None,
    intrinsic_reward: bool = False,
    intrinsic_reward_weight: float = 0.0,
    random_seed: int = 0,
    use_options: bool = False,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
) -> None:
    """Train a small RL agent from ``data_dir``."""
    set_seed(random_seed)

    if self_play:
        if SelfPlayEnv is None or train_self_play is None:
            raise ImportError("SelfPlayEnv not available for self-play training")
        env = SelfPlayEnv()
        trader_q, perturb_q = train_self_play(env, episodes=training_steps)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "trader_agent.npy", trader_q)
        np.save(out_dir / "perturbation_agent.npy", perturb_q)
        meta = {
            "model_id": "rl_agent_self_play",
            "training_type": "self_play",
            "training_steps": training_steps,
            "trader_agent": "trader_agent.npy",
            "perturbation_agent": "perturbation_agent.npy",
        }
        with open(out_dir / "model.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Self-play agents saved to {out_dir}")
        return

    states, actions, rewards_ext, next_states, vec = _build_dataset(
        data_dir, flight_uri, kafka_brokers
    )
    if use_options:
        train_options(
            states,
            actions,
            rewards_ext,
            out_dir,
            learning_rate=learning_rate,
            gamma=gamma,
            training_steps=training_steps,
            algo=algo,
            compress_model=compress_model,
            feature_names=vec.get_feature_names_out().tolist(),
        )
        return
    n_features = states.shape[1]

    rewards = rewards_ext.copy()
    drawdown = _max_drawdown(rewards_ext)
    var95 = _var_95(rewards_ext)
    if max_drawdown is not None or var_limit is not None:
        penalty = 0.0
        if max_drawdown is not None and drawdown > max_drawdown:
            penalty += drawdown - max_drawdown
        if var_limit is not None and var95 > var_limit:
            penalty += var95 - var_limit
        if penalty > 0:
            rewards = rewards - penalty
    risk_info = {
        "risk_params": {"max_drawdown": max_drawdown, "var_limit": var_limit},
        "risk_metrics": {"max_drawdown": drawdown, "var_95": var95},
    }
    if intrinsic_reward:
        if not HAS_SB3_CONTRIB:
            raise ImportError("sb3_contrib is required for intrinsic rewards")

        class _SimpleRND:
            def __init__(self, input_dim: int, lr: float = 1e-3, hidden: int = 32):
                self.target = np.random.randn(input_dim, hidden).astype(np.float32)
                self.predictor = np.random.randn(input_dim, hidden).astype(np.float32)
                self.lr = lr

            def bonus(self, obs: np.ndarray) -> float:
                obs = obs.astype(np.float32)
                t = obs @ self.target
                p = obs @ self.predictor
                err = t - p
                # simple SGD update
                self.predictor += self.lr * np.outer(obs, err)
                return float(np.mean(err**2))

        rnd = _SimpleRND(n_features)
        bonuses = [rnd.bonus(s) for s in states[: len(rewards)]]
        rewards = rewards + intrinsic_reward_weight * np.asarray(bonuses)

    init_model_data = None
    if start_model is not None and start_model.exists():
        try:
            with open(start_model) as f:
                init_model_data = json.load(f)
        except Exception:
            init_model_data = None

    algo_key = algo.lower().replace("-", "_")

    if self_play and algo_key not in {"ppo", "dqn", "c51", "qr_dqn", "a2c", "ddpg"}:
        raise ValueError(
            "--self-play requires ppo, dqn, c51, qr_dqn, a2c, or ddpg algorithm"
        )

    # precompute experience tuples for offline algorithms
    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = [
        (states[i], int(actions[i]), float(rewards[i]), next_states[i])
        for i in range(len(actions))
    ]

    buffer_client = (
        FederatedBufferClient(federated_server)
        if federated_server and FederatedBufferClient is not None
        else None
    )
    if buffer_client is not None:
        experiences = buffer_client.sync(experiences)

    if algo_key == "decision_transformer":
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch are required for decision_transformer"
            )

        seq_len = min(len(states), 20)
        state_tensor = torch.tensor(states[:seq_len], dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(actions[:seq_len], dtype=torch.long)
        returns = np.cumsum(rewards[::-1])[::-1][:seq_len]
        rtg_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)
        timestep_tensor = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        action_in = (
            torch.nn.functional.one_hot(action_tensor, num_classes=2)
            .float()
            .unsqueeze(0)
        )

        config = DecisionTransformerConfig(
            state_dim=n_features,
            act_dim=2,
            max_ep_len=seq_len,
            hidden_size=n_features,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        model = DecisionTransformerModel(config)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(training_steps):
            optim.zero_grad()
            out = model(
                states=state_tensor,
                actions=action_in,
                returns_to_go=rtg_tensor,
                timesteps=timestep_tensor,
            )
            logits = out.action_preds
            loss = loss_fn(logits.view(-1, 2), action_tensor.view(-1))
            loss.backward()
            optim.step()

        with torch.no_grad():
            out = model(
                states=state_tensor,
                actions=action_in,
                returns_to_go=rtg_tensor,
                timesteps=timestep_tensor,
            )
        preds = out.action_preds.argmax(dim=-1).view(-1).cpu().numpy()
        train_acc = float(np.mean(preds == actions[:seq_len]))

        block = model.encoder.h[0]
        qkv_w = block.attn.c_attn.weight.detach().cpu().numpy()
        qkv_b = block.attn.c_attn.bias.detach().cpu().numpy()
        qk = qkv_w[:, :n_features].reshape(-1).tolist()
        qb = qkv_b[:n_features].tolist()
        kk = qkv_w[:, n_features : 2 * n_features].reshape(-1).tolist()
        kb = qkv_b[n_features : 2 * n_features].tolist()
        vk = qkv_w[:, 2 * n_features :].reshape(-1).tolist()
        vb = qkv_b[2 * n_features :].tolist()
        out_w = block.attn.c_proj.weight.detach().cpu().numpy().reshape(-1).tolist()
        out_b = block.attn.c_proj.bias.detach().cpu().numpy().tolist()
        head_w = model.predict_action[0].weight.detach().cpu().numpy()
        head_b = model.predict_action[0].bias.detach().cpu().numpy()
        dense_w = (head_w[1] - head_w[0]).reshape(-1).tolist()
        dense_b = float(head_b[1] - head_b[0])
        weights = [qk, qb, kk, kb, vk, vb, out_w, out_b, dense_w, dense_b]

        out_dir.mkdir(parents=True, exist_ok=True)
        model_info = {
            "model_id": "rl_agent_dt",
            "algo": algo_key,
            "rl_algo": algo_key,
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": vec.get_feature_names_out().tolist(),
            "sequence_length": int(seq_len),
            "transformer_weights": weights,
            "feature_mean": states.mean(axis=0).astype(float).tolist(),
            "feature_std": states.std(axis=0).astype(float).tolist(),
            "train_accuracy": train_acc,
            "avg_reward": float(np.mean(rewards[:seq_len])),
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "val_accuracy": float("nan"),
            "accuracy": float("nan"),
            "num_samples": len(actions),
        }
        model_info.update(risk_info)
        model_info["training_type"] = "offline_rl"

        model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
        open_func = gzip.open if compress_model else open
        with open_func(model_path, "wt") as f:
            json.dump(model_info, f, indent=2)

        print(f"Model written to {model_path}")
        return

    if algo_key in {"ppo", "dqn", "c51", "qr_dqn", "a2c", "ddpg"}:
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is not installed")
        if algo_key in {"c51", "qr_dqn"} and not HAS_SB3_CONTRIB:
            raise ImportError("sb3-contrib is required for c51 or qr_dqn")
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
                obs = self.observations[self.idx] if not done else self.observations[-1]
                return obs, reward, done, False, {}

        def make_trade_env() -> Env:
            return TradeEnv(states, rewards)

        vec_cls = SubprocVecEnv if num_envs > 1 else None
        env = make_vec_env(make_trade_env, n_envs=num_envs, vec_env_cls=vec_cls)
        eval_env = TradeEnv(states, rewards_ext)
        algo_map = {"ppo": sb3.PPO, "dqn": sb3.DQN}
        if HAS_SB3:
            algo_map.update({"a2c": sb3.A2C, "ddpg": sb3.DDPG})
        if HAS_SB3_CONTRIB:
            algo_map.update({"c51": sb3c.C51, "qr_dqn": sb3c.QRDQN})
        model_cls = algo_map[algo_key]
        replay_args: Dict = {}
        if algo_key in {"dqn", "c51", "qr_dqn"}:
            buffer_kwargs: Dict[str, float | int] = {}
            if HAS_PRB:
                replay_args["replay_buffer_class"] = PrioritizedReplayBuffer
                buffer_kwargs.update({"alpha": replay_alpha, "beta": replay_beta})
                replay_args["learning_starts"] = 0
                if n_step > 1:
                    buffer_kwargs["n_step"] = n_step
            replay_args["replay_buffer_kwargs"] = buffer_kwargs
        model = model_cls(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=0,
            **replay_args,
        )
        if self_play:
            sim_env = SelfPlayEnv()
            half = max(1, training_steps // 2)
            model.learn(total_timesteps=half)
            model.set_env(sim_env)
            model.learn(total_timesteps=training_steps - half)
        else:
            model.learn(total_timesteps=training_steps)
        try:
            train_metrics = model.logger.get_log_dict()  # type: ignore[attr-defined]
        except Exception:
            train_metrics = {}
        if metrics_url:
            updates = _send_live_metrics(metrics_url, train_metrics)
            if isinstance(updates, dict):
                extra_steps = int(updates.get("extra_steps", 0))
                lr_new = updates.get("learning_rate")
                if isinstance(lr_new, (int, float)) and hasattr(model, "lr_schedule"):
                    model.lr_schedule = lambda _: lr_new
                if extra_steps > 0:
                    model.learn(total_timesteps=extra_steps)
        if self_play:
            model.set_env(env)
        preds: List[int] = []
        obs, _ = eval_env.reset()
        obs = obs.reshape(1, -1)
        for i in range(len(actions)):
            with tracer.start_as_current_span("decision") as dspan:
                act, _ = model.predict(obs, deterministic=True)
                act_i = int(act[0]) if np.ndim(act) > 0 else int(act)
                preds.append(act_i)
                obs, reward, done, _, _ = eval_env.step(act_i)
                obs = obs.reshape(1, -1)
                dctx = dspan.get_span_context()
                logger.info(
                    {"decision_id": i, "action": act_i, "reward": float(reward)},
                    extra={"trace_id": dctx.trace_id, "span_id": dctx.span_id},
                )
                if done:
                    break
        train_acc = float(np.mean(np.array(preds) == np.array(actions)))

        # Evaluate episodic rewards for each environment
        eval_vec_env = make_vec_env(
            make_trade_env, n_envs=num_envs, vec_env_cls=vec_cls
        )
        vec_obs = eval_vec_env.reset()
        dones = np.zeros(num_envs, dtype=bool)
        episode_rewards = np.zeros(num_envs, dtype=float)
        while not np.all(dones):
            acts, _ = model.predict(vec_obs, deterministic=True)
            vec_obs, rewards_step, dones, _ = eval_vec_env.step(acts)
            episode_rewards += rewards_step
        ctx_env = trace.get_current_span().get_span_context()
        for idx, r in enumerate(episode_rewards.tolist()):
            logger.info(
                {"env_id": idx, "episode_reward": float(r)},
                extra={"trace_id": ctx_env.trace_id, "span_id": ctx_env.span_id},
            )
        avg_ep_reward = float(np.mean(episode_rewards))
        eval_vec_env.close()

        expected_return = float("nan")
        downside_risk = float("nan")
        value_atoms: List[float] | None = None
        value_dist: List[float] | None = None
        value_quantiles: List[float] | None = None
        value_mean: float | None = None
        value_std: float | None = None
        if HAS_SB3_CONTRIB and algo_key in {"c51", "qr_dqn"} and torch is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(states, dtype=torch.float32)
                dist = model.policy.q_net(obs_tensor)
                if algo_key == "c51":
                    probs = torch.softmax(dist, dim=-1).cpu().numpy()
                    atoms = model.policy.support.cpu().numpy()
                    q_vals = (probs * atoms).sum(-1)
                    best = q_vals.argmax(-1)
                    exp_returns = q_vals[np.arange(len(q_vals)), best]
                    downside = [
                        probs[i, a][atoms < 0].sum() for i, a in enumerate(best)
                    ]
                    value_atoms = atoms.tolist()
                    value_dist = (
                        probs[np.arange(len(q_vals)), best].mean(axis=0).tolist()
                    )
                    _atoms = np.array(value_atoms, dtype=float)
                    _dist = np.array(value_dist, dtype=float)
                    value_mean = float(np.dot(_atoms, _dist))
                    value_std = float(
                        np.sqrt(((_atoms - value_mean) ** 2 * _dist).sum())
                    )
                else:
                    quantiles = dist.cpu().numpy()
                    q_vals = quantiles.mean(-1)
                    best = q_vals.argmax(-1)
                    exp_returns = q_vals[np.arange(len(q_vals)), best]
                    downside = [
                        (quantiles[i, a] < 0).mean() for i, a in enumerate(best)
                    ]
                    value_quantiles = (
                        quantiles[np.arange(len(q_vals)), best].mean(axis=0).tolist()
                    )
                    value_mean = float(np.mean(value_quantiles))
                    value_std = float(np.std(value_quantiles))
                expected_return = float(np.mean(exp_returns))
                downside_risk = float(np.mean(downside))
                ctx_log = trace.get_current_span().get_span_context()
                logger.info(
                    {
                        "expected_return": expected_return,
                        "downside_risk": downside_risk,
                    },
                    extra={"trace_id": ctx_log.trace_id, "span_id": ctx_log.span_id},
                )
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
            "avg_reward": avg_ep_reward,
            "episode_rewards": episode_rewards.tolist(),
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "gamma": gamma,
            "val_accuracy": float("nan"),
            "accuracy": float("nan"),
            "num_samples": len(actions),
            "weights_file": weights_path.with_suffix(".zip").name,
            "intrinsic_reward": intrinsic_reward,
        }
        model_info.update(risk_info)
        model_info["replay_alpha"] = replay_alpha
        model_info["replay_beta"] = replay_beta
        model_info["n_step"] = n_step
        if intrinsic_reward:
            model_info["intrinsic_reward_weight"] = intrinsic_reward_weight
        if not np.isnan(expected_return):
            model_info["expected_return"] = expected_return
        if not np.isnan(downside_risk):
            model_info["downside_risk"] = downside_risk
        model_info["train_metrics"] = train_metrics
        if value_atoms and value_dist:
            model_info["value_atoms"] = value_atoms
            model_info["value_distribution"] = value_dist
        if value_quantiles:
            model_info["value_quantiles"] = value_quantiles
        if value_mean is not None:
            model_info["value_mean"] = value_mean
        if value_std is not None:
            model_info["value_std"] = value_std
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
        feats_match = (
            init_model_data.get("feature_names") == vec.get_feature_names_out().tolist()
        )
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
    episode_td: List[float] = []
    if algo_key == "cql":
        alpha = 0.01  # conservative penalty strength
        for i in range(training_steps):
            if buffer_client is not None and i % sync_interval == 0:
                experiences = buffer_client.sync(experiences)
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
        episode_totals: List[float] = []  # total reward per episode
        for i in range(training_steps):
            if buffer_client is not None and i % sync_interval == 0:
                experiences = buffer_client.sync(experiences)
            total_r = 0.0
            td_errs: List[float] = []
            if HAS_PRB:
                obs_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
                )
                action_space = spaces.Discrete(2)
                pr_buffer = PrioritizedReplayBuffer(
                    buffer_size, obs_space, action_space, alpha=replay_alpha
                )
            else:
                buffer: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
                max_prio = 1.0
            for step, exp in enumerate(experiences):
                s, a, r, ns = exp
                total_r += r
                if HAS_PRB:
                    pr_buffer.add(s, ns, np.array([a]), np.array([r]), np.array([0.0]))
                else:
                    if len(buffer) >= buffer_size:
                        buffer.pop(0)
                    buffer.append((s, a, r, ns, max_prio))

                if step % update_freq == 0:
                    if HAS_PRB and pr_buffer.size() >= batch_size:
                        sample = pr_buffer.sample(batch_size, beta=replay_beta)
                        for j in range(batch_size):
                            bs = sample.observations[j]
                            ba = int(sample.actions[j])
                            br = float(sample.rewards[j])
                            bns = sample.next_observations[j]
                            w = float(sample.weights[j])
                            q_next0 = intercepts[0] + np.dot(weights[0], bns)
                            q_next1 = intercepts[1] + np.dot(weights[1], bns)
                            q_target = br + gamma * max(q_next0, q_next1)
                            q_current = intercepts[ba] + np.dot(weights[ba], bs)
                            td_err = q_target - q_current
                            weights[ba] += learning_rate * w * td_err * bs
                            intercepts[ba] += learning_rate * w * td_err
                            pr_buffer.update_priorities(
                                [sample.indices[j]], np.array([abs(td_err) + 1e-6])
                            )
                            td_errs.append(abs(td_err))
                    elif not HAS_PRB and len(buffer) >= batch_size:
                        weights_arr = np.array([b[4] for b in buffer], dtype=float)
                        scaled = weights_arr**replay_alpha
                        prob = scaled / scaled.sum()
                        batch_idx = np.random.choice(
                            len(buffer), size=batch_size, p=prob
                        )
                        max_w = (len(buffer) * prob.min()) ** (-replay_beta)
                        for idx in batch_idx:
                            bs, ba, br, bns, bw = buffer[idx]
                            q_next0 = intercepts[0] + np.dot(weights[0], bns)
                            q_next1 = intercepts[1] + np.dot(weights[1], bns)
                            q_target = br + gamma * max(q_next0, q_next1)
                            q_current = intercepts[ba] + np.dot(weights[ba], bs)
                            td_err = q_target - q_current
                            w = (len(buffer) * prob[idx]) ** (-replay_beta)
                            w /= max_w
                            weights[ba] += learning_rate * w * td_err * bs
                            intercepts[ba] += learning_rate * w * td_err
                            pr = float(abs(td_err)) + 1e-6
                            buffer[idx] = (bs, ba, br, bns, pr)
                            if pr > max_prio:
                                max_prio = pr
                            td_errs.append(abs(td_err))
            episode_rewards.append(total_r / len(experiences))
            episode_totals.append(total_r)
            episode_td.append(float(np.mean(td_errs)) if td_errs else 0.0)
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
        "gamma": gamma,
        "epsilon": epsilon,
        "replay_alpha": replay_alpha,
        "replay_beta": replay_beta,
        "val_accuracy": float("nan"),
        "accuracy": float("nan"),
        "num_samples": len(actions),
        "avg_td_error": float(np.mean(episode_td)) if episode_td else 0.0,
        "td_errors": [float(e) for e in episode_td],
    }
    model.update(risk_info)
    model["training_type"] = training_type
    if training_type != "offline_rl" and init_model_data is not None:
        model["init_model"] = start_model.name if start_model is not None else None
        model["init_model_id"] = init_model_data.get("model_id")

    model.setdefault("metadata", {})["seed"] = random_seed
    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {model_path}")


def main() -> None:
    with tracer.start_as_current_span("train_rl_agent") as span:
        ctx = span.get_span_context()
        logger.info(
            "start training", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id}
        )
        p = argparse.ArgumentParser(description="Train RL agent from logs")
        p.add_argument("--data-dir", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--flight-uri", help="Arrow Flight server URI")
        p.add_argument("--kafka-brokers", help="Kafka bootstrap servers for log replay")
        p.add_argument("--learning-rate", type=float, default=0.1, help="learning rate")
        p.add_argument(
            "--epsilon", type=float, default=0.1, help="epsilon for exploration"
        )
        p.add_argument("--gamma", type=float, default=0.9, help="discount factor")
        p.add_argument(
            "--training-steps", type=int, default=100, help="total training steps"
        )
        p.add_argument(
            "--batch-size", type=int, default=4, help="batch size for updates"
        )
        p.add_argument(
            "--buffer-size", type=int, default=100, help="replay buffer size"
        )
        p.add_argument(
            "--update-freq", type=int, default=1, help="steps between updates"
        )
        p.add_argument(
            "--replay-alpha",
            type=float,
            default=0.6,
            help="prioritized replay exponent",
        )
        p.add_argument(
            "--replay-beta", type=float, default=0.4, help="IS weight exponent"
        )
        p.add_argument("--n-step", type=int, default=1, help="n-step return length")
        p.add_argument(
            "--num-envs", type=int, default=1, help="number of parallel environments"
        )
        p.add_argument(
            "--algo",
            default="dqn",
            choices=[
                "dqn",
                "ppo",
                "a2c",
                "ddpg",
                "c51",
                "qr_dqn",
                "qlearn",
                "cql",
                "decision_transformer",
            ],
            help=(
                "RL algorithm: dqn (default), ppo, a2c, ddpg, c51, qr_dqn, qlearn, cql, or "
                "decision_transformer (requires transformers)."
            ),
        )
        p.add_argument("--start-model", help="path to initial model coefficients")
        p.add_argument(
            "--compress-model", action="store_true", help="write model.json.gz"
        )
        p.add_argument(
            "--self-play",
            action="store_true",
            help="alternate training on real logs and simulated data",
        )
        p.add_argument(
            "--federated-server", help="gRPC address of federated buffer server"
        )
        p.add_argument(
            "--sync-interval",
            type=int,
            default=50,
            help="training iterations between federated buffer syncs",
        )
        p.add_argument(
            "--metrics-url", help="endpoint to POST live metrics for feedback"
        )
        p.add_argument(
            "--intrinsic-reward",
            action="store_true",
            help="enable intrinsic reward bonus (requires sb3-contrib)",
        )
        p.add_argument(
            "--intrinsic-weight",
            type=float,
            default=0.0,
            help="weight for intrinsic reward bonus",
        )
        p.add_argument("--random-seed", type=int, default=0)
        args = p.parse_args()
        train(
            Path(args.data_dir),
            Path(args.out_dir),
            learning_rate=args.learning_rate,
            epsilon=args.epsilon,
            gamma=args.gamma,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            update_freq=args.update_freq,
            algo=args.algo,
            replay_alpha=args.replay_alpha,
            replay_beta=args.replay_beta,
            n_step=args.n_step,
            num_envs=args.num_envs,
            start_model=Path(args.start_model) if args.start_model else None,
            compress_model=args.compress_model,
            self_play=args.self_play,
            flight_uri=args.flight_uri,
            kafka_brokers=args.kafka_brokers,
            federated_server=args.federated_server,
            sync_interval=args.sync_interval,
            metrics_url=args.metrics_url,
            intrinsic_reward=args.intrinsic_reward,
            intrinsic_reward_weight=args.intrinsic_weight,
            random_seed=args.random_seed,
        )
        logger.info(
            "training complete",
            extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id},
        )


if __name__ == "__main__":
    main()
