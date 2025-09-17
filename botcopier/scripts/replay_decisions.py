#!/usr/bin/env python3
"""Replay stored decision logs with a new model.

The script loads a decision log produced by the trading bot along with a
new ``model.json`` file. It re-computes probabilities for each decision
using the new model and reports any divergences between the original and
replayed outcomes. Summary statistics such as accuracy and the change in
profit are printed at the end.

Any mismatched decisions are written to ``divergences.csv`` and can
optionally be tagged with a sample ``weight``. This file can then be fed
back into :mod:`botcopier.training.pipeline` via ``--replay-file`` to
emphasise corrections during the next round of training.

When sufficient hardware resources are available, ``detect_resources`` is
used to determine whether a more complex neural network representation
should be used instead of the basic logistic regression.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from botcopier.data.schema import DECISION_LOG_SCHEMA

try:  # optional torch dependency for encoder
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

if _HAS_TORCH:
    from botcopier.models.deep import MixtureOfExperts, TabTransformer, TCNClassifier
else:  # pragma: no cover - optional dependency
    TabTransformer = None  # type: ignore
    TCNClassifier = None  # type: ignore
    MixtureOfExperts = None  # type: ignore
from botcopier.training.pipeline import detect_resources

try:  # optional graph embedding support
    from graph_dataset import GraphDataset, compute_gnn_embeddings

    _HAS_TG = True
except Exception:  # pragma: no cover
    GraphDataset = None  # type: ignore
    compute_gnn_embeddings = None  # type: ignore
    _HAS_TG = False

try:  # optional stable-baselines3 dependency
    import stable_baselines3 as sb3  # type: ignore

    from botcopier.rl.options import (
        OptionTradeEnv,
        default_skills,
        evaluate_option_policy,
    )

    _HAS_SB3 = True
except Exception:  # pragma: no cover - optional dependency
    sb3 = None  # type: ignore
    OptionTradeEnv = None  # type: ignore
    default_skills = None  # type: ignore
    evaluate_option_policy = None  # type: ignore
    _HAS_SB3 = False


def _load_model(model_file: Path) -> Dict:
    """Load model parameters from ``model_file``."""
    open_func = gzip.open if model_file.suffix == ".gz" else open
    with open_func(model_file, "rt") as f:
        return json.load(f)


def _predict_logistic(model: Dict, features: Dict[str, float]) -> float:
    """Compute probability using a logistic regression model."""
    names = model.get("feature_names", [])
    coeffs = np.array(model.get("coefficients", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))
    mean = np.array(
        model.get("feature_mean", model.get("mean", [0.0] * len(names))),
        dtype=float,
    )
    std = np.array(
        model.get("feature_std", model.get("std", [1.0] * len(names))),
        dtype=float,
    )
    low = np.array(model.get("clip_low", [-np.inf] * len(names)), dtype=float)
    high = np.array(model.get("clip_high", [np.inf] * len(names)), dtype=float)

    vec = np.array([float(features.get(n, 0.0)) for n in names])
    vec = np.clip(vec, low, high)
    if vec.shape[0] != len(coeffs):
        coeffs = coeffs[: vec.shape[0]]
    std_safe = np.where(std == 0, 1, std)
    z = ((vec - mean) / std_safe) @ coeffs + intercept
    prob = float(1 / (1 + math.exp(-z)))
    calib = model.get("calibration")
    if isinstance(calib, dict):
        if calib.get("method") == "isotonic":
            x = np.asarray(calib.get("x", []), dtype=float)
            y = np.asarray(calib.get("y", []), dtype=float)
            if x.size and y.size:
                prob = float(np.interp(prob, x, y, left=y[0], right=y[-1]))
        elif {"coef", "intercept"} <= calib.keys():
            z = z * float(calib["coef"]) + float(calib["intercept"])
            prob = float(1 / (1 + math.exp(-z)))
    else:
        cal_coef = model.get("calibration_coef")
        cal_inter = model.get("calibration_intercept")
        if cal_coef is not None and cal_inter is not None:
            z = z * float(cal_coef) + float(cal_inter)
            prob = float(1 / (1 + math.exp(-z)))
    return prob


def _predict_nn(model: Dict, features: Dict[str, float]) -> float:
    """Compute probability using a simple neural network if available."""
    weights = model.get("nn_weights")
    if not weights:
        return _predict_logistic(model, features)
    names = model.get("feature_names", [])
    mean = np.array(
        model.get("feature_mean", model.get("mean", [0.0] * len(names))),
        dtype=float,
    )
    std = np.array(
        model.get("feature_std", model.get("std", [1.0] * len(names))),
        dtype=float,
    )
    low = np.array(model.get("clip_low", [-np.inf] * len(names)), dtype=float)
    high = np.array(model.get("clip_high", [np.inf] * len(names)), dtype=float)
    vec = np.array([float(features.get(n, 0.0)) for n in names])
    vec = np.clip(vec, low, high)
    std_safe = np.where(std == 0, 1, std)
    x = (vec - mean) / std_safe
    l1_w, l1_b, l2_w, l2_b = [np.array(w, dtype=float) for w in weights[:4]]
    h = np.tanh(np.dot(x, l1_w) + l1_b)
    z = np.dot(h, l2_w) + l2_b
    return float(1 / (1 + np.exp(-z)))


def _prepare_sequence_window(
    model: Dict, features: Dict[str, float]
) -> tuple[np.ndarray | None, int]:
    names = model.get("feature_names", [])
    if not names:
        return None, 1
    mean = np.asarray(model.get("mean", [0.0] * len(names)), dtype=float)
    std = np.asarray(model.get("std", [1.0] * len(names)), dtype=float)
    low = np.asarray(model.get("clip_low", [-np.inf] * len(names)), dtype=float)
    high = np.asarray(model.get("clip_high", [np.inf] * len(names)), dtype=float)
    vec = np.array([float(features.get(n, 0.0)) for n in names], dtype=float)
    vec = np.clip(vec, low, high)
    std_safe = np.where(std == 0, 1, std)
    norm = (vec - mean) / std_safe
    window = int(model.get("window", len(model.get("sequence_order", [])) or 1))
    buffer = model.setdefault("_sequence_buffer", [])
    buffer.append(norm)
    if len(buffer) > window:
        del buffer[:-window]
    if len(buffer) < window:
        return None, window
    seq = np.stack(buffer[-window:], axis=0)
    return seq, window


def _load_sequence_model(model: Dict, cls):
    if not _HAS_TORCH:
        return None
    cache_key = f"_{cls.__name__.lower()}"
    net = model.get(cache_key)
    if net is not None:
        return net
    state = model.get("state_dict")
    names = model.get("feature_names", [])
    if not state or not names:
        return None
    window = int(model.get("window", len(model.get("sequence_order", [])) or 1))
    config = model.get("config", {})
    if cls is TabTransformer and TabTransformer is not None:
        net = TabTransformer(
            len(names),
            window,
            dim=int(config.get("dim", 64)),
            depth=int(config.get("depth", 2)),
            heads=int(config.get("heads", 4)),
            ff_dim=int(config.get("ff_dim", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )
    elif cls is TCNClassifier and TCNClassifier is not None:
        channels = config.get("channels")
        if channels is not None:
            channels = [int(c) for c in channels]
        net = TCNClassifier(
            len(names),
            window,
            channels=channels,
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.1)),
        )
    else:  # pragma: no cover - defensive
        return None
    state_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}
    net.load_state_dict(state_tensors)
    net.eval()
    model[cache_key] = net
    return net


def _load_moe_model(model: Dict):
    if not _HAS_TORCH or MixtureOfExperts is None:
        return None
    net = model.get("_mixtureofexperts")
    if net is not None:
        return net
    state = model.get("state_dict")
    feature_names = model.get("feature_names", [])
    gating = model.get("regime_gating", {})
    regime_features = gating.get("feature_names") or model.get("regime_features", [])
    if not state or not feature_names or not regime_features:
        return None
    arch = model.get("architecture", {})
    weights = gating.get("weights") or []
    experts_meta = model.get("experts") or []
    n_experts = int(
        arch.get("n_experts")
        or (len(weights) if isinstance(weights, list) else 0)
        or len(experts_meta)
    )
    dropout = float(arch.get("dropout", 0.0))
    net = MixtureOfExperts(len(feature_names), len(regime_features), n_experts, dropout=dropout)
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}
    net.load_state_dict(tensors)
    net.eval()
    model["_mixtureofexperts"] = net
    return net


def _predict_tabtransformer(model: Dict, features: Dict[str, float]) -> float:
    """Compute probability using a tabular transformer model."""
    if not _HAS_TORCH:
        return _predict_logistic(model, features)
    seq, _ = _prepare_sequence_window(model, features)
    if seq is None:
        return _predict_logistic(model, features)
    net = _load_sequence_model(model, TabTransformer)
    if net is None:
        return _predict_logistic(model, features)
    with torch.no_grad():
        tens = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        logit = net(tens)
        return float(torch.sigmoid(logit).item())


def _predict_tcn(model: Dict, features: Dict[str, float]) -> float:
    if not _HAS_TORCH:
        return _predict_logistic(model, features)
    seq, _ = _prepare_sequence_window(model, features)
    if seq is None:
        return _predict_logistic(model, features)
    net = _load_sequence_model(model, TCNClassifier)
    if net is None:
        return _predict_logistic(model, features)
    with torch.no_grad():
        tens = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        logit = net(tens)
        return float(torch.sigmoid(logit).item())


def _predict_moe(model: Dict, features: Dict[str, float]) -> float:
    feature_names = model.get("feature_names", [])
    gating = model.get("regime_gating", {})
    regime_names = gating.get("feature_names") or model.get("regime_features", [])
    if not feature_names or not regime_names:
        return _predict_logistic(model, features)
    base_vec = np.array([float(features.get(n, 0.0)) for n in feature_names], dtype=float)
    regime_vec = np.array([float(features.get(n, 0.0)) for n in regime_names], dtype=float)
    if _HAS_TORCH:
        net = _load_moe_model(model)
        if net is not None:
            with torch.no_grad():
                base_t = torch.tensor(base_vec, dtype=torch.float32).unsqueeze(0)
                regime_t = torch.tensor(regime_vec, dtype=torch.float32).unsqueeze(0)
                prob, _ = net(base_t, regime_t)
                return float(prob.squeeze(0).item())
    experts = model.get("experts", [])
    weights = np.asarray(gating.get("weights", []), dtype=float)
    bias = np.asarray(gating.get("bias", []), dtype=float)
    if not len(experts) or weights.size == 0 or bias.size == 0:
        return _predict_logistic(model, features)
    expert_logits = np.array(
        [
            base_vec @ np.asarray(exp.get("weights", []), dtype=float)
            + float(exp.get("bias", 0.0))
            for exp in experts
        ],
        dtype=float,
    )
    expert_prob = 1.0 / (1.0 + np.exp(-expert_logits))
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    gate_logits = regime_vec @ weights.T + bias
    gate_logits = np.asarray(gate_logits, dtype=float).ravel()
    if gate_logits.size == 0:
        return _predict_logistic(model, features)
    gate_logits = gate_logits - np.max(gate_logits)
    gate_exp = np.exp(gate_logits)
    if not np.isfinite(gate_exp).all():
        gate_exp = np.ones_like(gate_exp)
    gate = gate_exp / gate_exp.sum()
    return float(np.dot(gate, expert_prob))


def _load_logs(log_file: Path) -> pd.DataFrame:
    """Load decision logs from ``log_file``."""
    table = pq.read_table(log_file, schema=DECISION_LOG_SCHEMA)
    df = table.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if "event_id" in df.columns and "decision_id" not in df.columns:
        df.rename(columns={"event_id": "decision_id"}, inplace=True)
    return df


def _recompute(
    df: pd.DataFrame, model: Dict, threshold: float, model_dir: Path | None = None
) -> Dict:
    """Recompute probabilities and collect statistics."""
    resources = detect_resources()
    model_type = model.get("model_type")
    if model_type in {"tabtransformer", "transformer"}:
        pred_fn = _predict_tabtransformer
    elif model_type == "tcn":
        pred_fn = _predict_tcn
    elif model_type == "moe":
        pred_fn = _predict_moe
    elif model.get("state_dict"):
        pred_fn = _predict_tabtransformer
    else:
        use_complex = not resources.get("lite_mode") and model.get("nn_weights")
        pred_fn = _predict_nn if use_complex else _predict_logistic

    gnn_state = model.get("gnn_state")
    if (
        _HAS_TG
        and gnn_state
        and Path("symbol_graph.json").exists()
        and "symbol" in df.columns
    ):
        try:
            dataset = GraphDataset(Path("symbol_graph.json"))
            emb_map, _ = compute_gnn_embeddings(df, dataset, state_dict=gnn_state)
            if emb_map:
                emb_dim = len(next(iter(emb_map.values())))
                sym_series = df["symbol"].astype(str)
                for i in range(emb_dim):
                    col = f"graph_emb{i}"
                    df[col] = sym_series.map(
                        lambda s: emb_map.get(s, [0.0] * emb_dim)[i]
                    )
        except Exception:
            pass

    enc_meta = model.get("encoder")
    if enc_meta and _HAS_TORCH:
        enc_file = Path(enc_meta.get("file", "encoder.pt"))
        if not enc_file.is_absolute() and model_dir is not None:
            enc_file = model_dir / enc_file
        tick_cols = [c for c in df.columns if c.startswith("tick_")]
        if tick_cols and enc_file.exists():
            tick_cols = sorted(
                tick_cols,
                key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0,
            )
            try:
                state = torch.load(enc_file, map_location="cpu")
                weight = state.get("state_dict", {}).get("weight")
                if weight is not None:
                    weight_t = weight.float().t()
                    window = weight_t.shape[0]
                    cols = tick_cols[:window]
                    if len(cols) == window:
                        X = torch.tensor(
                            df[cols].to_numpy(dtype=float), dtype=torch.float32
                        )
                        emb = X @ weight_t
                        for i in range(weight_t.shape[1]):
                            df[f"enc_{i}"] = emb[:, i].numpy()
            except Exception:
                pass

    divergences = []
    profits = df.get("profit")
    old_probs = df.get("probability")
    if old_probs is None:
        old_probs = df.get("prob")

    def features_from_row(row: pd.Series) -> Dict[str, float]:
        feat = {k: row.get(k, 0.0) for k in model.get("feature_names", [])}
        gating = model.get("regime_gating", {})
        regime_names = gating.get("feature_names") or model.get("regime_features", [])
        for name in regime_names:
            feat.setdefault(name, row.get(name, 0.0))
        return feat

    new_probs = []
    for _, row in df.iterrows():
        prob = pred_fn(model, features_from_row(row))
        new_probs.append(prob)
        if old_probs is not None:
            old_p = float(row.get(old_probs.name, 0.0))
            if (old_p >= threshold) != (prob >= threshold):
                divergences.append(
                    {
                        "decision_id": row.get("decision_id"),
                        "old_prob": old_p,
                        "new_prob": prob,
                        "profit": row.get("profit", 0.0),
                    }
                )
    df["new_probability"] = new_probs

    actual = (df.get("profit", 0.0) > 0).astype(int)
    old_pred = (
        (old_probs >= threshold).astype(int)
        if old_probs is not None
        else pd.Series([1] * len(df))
    )
    new_pred = (df["new_probability"] >= threshold).astype(int)

    accuracy_old = float((old_pred == actual).mean())
    accuracy_new = float((new_pred == actual).mean())

    profit_delta = 0.0
    if profits is not None and old_probs is not None:
        mask_skip = (old_pred == 1) & (new_pred == 0)
        profit_delta = float(-profits[mask_skip].sum())

    tp = int(((new_pred == 1) & (actual == 1)).sum())
    fp = int(((new_pred == 1) & (actual == 0)).sum())
    tn = int(((new_pred == 0) & (actual == 0)).sum())
    fn = int(((new_pred == 0) & (actual == 1)).sum())

    return {
        "divergences": divergences,
        "accuracy_old": accuracy_old,
        "accuracy_new": accuracy_new,
        "profit_delta": profit_delta,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def replay_option_policy(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    model: Dict,
    model_dir: Path,
) -> Dict:
    """Replay decisions using an option policy saved in ``model``."""

    if not _HAS_SB3 or sb3 is None or OptionTradeEnv is None or default_skills is None:
        raise ImportError("stable-baselines3 is required for option policy replay")

    skills = default_skills()
    env = OptionTradeEnv(states, actions, rewards, skills)
    weights_file = model_dir / model.get("option_weights_file", "")
    policy = sb3.PPO.load(str(weights_file))
    total_reward = evaluate_option_policy(policy, env)
    return {"total_reward": float(total_reward)}


def main() -> int:
    p = argparse.ArgumentParser(description="Replay decision logs with a new model")
    p.add_argument(
        "log_file",
        type=Path,
        nargs="?",
        default=Path("decisions.csv"),
        help="CSV decision log (default decisions.csv)",
    )
    p.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=Path("model.json"),
        help="Path to model.json or model.json.gz (default model.json)",
    )
    p.add_argument("--threshold", type=float, help="Override model threshold")
    p.add_argument(
        "--max-divergences",
        type=int,
        default=20,
        help="Show at most this many divergences",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("divergences.csv"),
        help="CSV file to write divergent decisions (default divergences.csv)",
    )
    p.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="sample weight assigned to each divergent trade",
    )
    args = p.parse_args()

    model = _load_model(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(model.get("threshold", 0.5))
    )
    df = _load_logs(args.log_file)
    stats = _recompute(df, model, threshold, args.model.parent)

    if model.get("options") and model.get("option_weights_file"):
        try:
            state_cols = model.get("feature_names", [])
            states = df[state_cols].to_numpy(dtype=float)
            acts = df.get("action", pd.Series([0] * len(df))).to_numpy(dtype=int)
            rewards = np.ones(len(df), dtype=float)
            opt_stats = replay_option_policy(
                states, acts, rewards, model, args.model.parent
            )
            print(f"Option total reward: {opt_stats['total_reward']:.2f}")
        except Exception:
            pass

    if stats["divergences"]:
        out_df = pd.DataFrame(stats["divergences"])
        out_df["weight"] = args.weight
        out_df.to_csv(args.output, index=False)

    print(f"Old accuracy: {stats['accuracy_old']:.3f}")
    print(f"New accuracy: {stats['accuracy_new']:.3f}")
    print(f"Profit delta: {stats['profit_delta']:.2f}")
    print(
        f"Confusion matrix: TP={stats['tp']} FP={stats['fp']} TN={stats['tn']} FN={stats['fn']}"
    )
    if stats["divergences"]:
        print("Divergent decisions:")
        for d in stats["divergences"][: args.max_divergences]:
            print(
                f"decision {d['decision_id']}: old={d['old_prob']:.3f} new={d['new_prob']:.3f} profit={d['profit']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
