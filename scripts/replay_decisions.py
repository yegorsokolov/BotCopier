#!/usr/bin/env python3
"""Replay stored decision logs with a new model.

The script loads a decision log produced by the trading bot along with a
new ``model.json`` file. It re-computes probabilities for each decision
using the new model and reports any divergences between the original and
replayed outcomes. Summary statistics such as accuracy and the change in
profit are printed at the end.

When sufficient hardware resources are available, ``detect_resources`` is
used to determine whether a more complex neural network representation
should be used instead of the basic logistic regression.
"""

from __future__ import annotations

import argparse
import json
import gzip
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from train_target_clone import detect_resources


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
    mean = np.array(model.get("mean", [0.0] * len(names)), dtype=float)
    std = np.array(model.get("std", [1.0] * len(names)), dtype=float)

    vec = np.array([float(features.get(n, 0.0)) for n in names])
    if vec.shape[0] != len(coeffs):
        coeffs = coeffs[: vec.shape[0]]
    std_safe = np.where(std == 0, 1, std)
    z = ((vec - mean) / std_safe) @ coeffs + intercept
    cal_coef = model.get("calibration_coef")
    cal_inter = model.get("calibration_intercept")
    if cal_coef is not None and cal_inter is not None:
        z = z * float(cal_coef) + float(cal_inter)
    return float(1 / (1 + math.exp(-z)))


def _predict_nn(model: Dict, features: Dict[str, float]) -> float:
    """Compute probability using a simple neural network if available."""
    weights = model.get("nn_weights")
    if not weights:
        return _predict_logistic(model, features)
    names = model.get("feature_names", [])
    mean = np.array(model.get("mean", [0.0] * len(names)), dtype=float)
    std = np.array(model.get("std", [1.0] * len(names)), dtype=float)
    vec = np.array([float(features.get(n, 0.0)) for n in names])
    std_safe = np.where(std == 0, 1, std)
    x = (vec - mean) / std_safe
    l1_w, l1_b, l2_w, l2_b = [np.array(w, dtype=float) for w in weights[:4]]
    h = np.tanh(np.dot(x, l1_w) + l1_b)
    z = np.dot(h, l2_w) + l2_b
    return float(1 / (1 + np.exp(-z)))


def _load_logs(log_file: Path) -> pd.DataFrame:
    """Load decision logs from ``log_file``."""
    df = pd.read_csv(log_file, sep=";")
    df.columns = [c.lower() for c in df.columns]
    return df


def _recompute(df: pd.DataFrame, model: Dict, threshold: float) -> Dict:
    """Recompute probabilities and collect statistics."""
    resources = detect_resources()
    use_complex = not resources.get("lite_mode") and model.get("nn_weights")
    pred_fn = _predict_nn if use_complex else _predict_logistic

    divergences = []
    profits = df.get("profit")
    old_probs = df.get("probability") or df.get("prob")

    def features_from_row(row: pd.Series) -> Dict[str, float]:
        return {k: row.get(k, 0.0) for k in model.get("feature_names", [])}

    new_probs = []
    for _, row in df.iterrows():
        prob = pred_fn(model, features_from_row(row))
        new_probs.append(prob)
        if old_probs is not None:
            old_p = float(row.get(old_probs.name, 0.0))
            if (old_p >= threshold) != (prob >= threshold):
                divergences.append(
                    {
                        "event_id": row.get("event_id"),
                        "old_prob": old_p,
                        "new_prob": prob,
                        "profit": row.get("profit", 0.0),
                    }
                )
    df["new_probability"] = new_probs

    actual = (df.get("profit", 0.0) > 0).astype(int)
    old_pred = (old_probs >= threshold).astype(int) if old_probs is not None else pd.Series([1] * len(df))
    new_pred = (df["new_probability"] >= threshold).astype(int)

    accuracy_old = float((old_pred == actual).mean())
    accuracy_new = float((new_pred == actual).mean())

    profit_delta = 0.0
    if profits is not None and old_probs is not None:
        mask_skip = (old_pred == 1) & (new_pred == 0)
        profit_delta = float(-profits[mask_skip].sum())

    return {
        "divergences": divergences,
        "accuracy_old": accuracy_old,
        "accuracy_new": accuracy_new,
        "profit_delta": profit_delta,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Replay decision logs with a new model")
    p.add_argument("log_file", type=Path, help="CSV decision log")
    p.add_argument("model", type=Path, help="Path to model.json or model.json.gz")
    p.add_argument("--threshold", type=float, help="Override model threshold")
    p.add_argument(
        "--max-divergences", type=int, default=20, help="Show at most this many divergences"
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write divergent decisions for further training",
    )
    args = p.parse_args()

    model = _load_model(args.model)
    threshold = args.threshold if args.threshold is not None else float(model.get("threshold", 0.5))
    df = _load_logs(args.log_file)
    stats = _recompute(df, model, threshold)

    if args.output and stats["divergences"]:
        pd.DataFrame(stats["divergences"]).to_csv(args.output, index=False)

    print(f"Old accuracy: {stats['accuracy_old']:.3f}")
    print(f"New accuracy: {stats['accuracy_new']:.3f}")
    print(f"Profit delta: {stats['profit_delta']:.2f}")
    if stats["divergences"]:
        print("Divergent decisions:")
        for d in stats["divergences"][: args.max_divergences]:
            print(
                f"event {d['event_id']}: old={d['old_prob']:.3f} new={d['new_prob']:.3f} profit={d['profit']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
