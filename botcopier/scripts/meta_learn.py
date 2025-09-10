"""Meta-learning utilities for symbol-based tasks.

This module provides a helper to sample tasks grouped by trading symbol and
train a shared initialisation using the Reptile meta learning algorithm.  The
resulting weights can optionally be written to ``model.json`` so that other
training scripts can start from a better initialisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List

import numpy as np
import pandas as pd

from .meta_adapt import ReptileMetaLearner


# ---------------------------------------------------------------------------
# Task sampling
# ---------------------------------------------------------------------------

def sample_symbol_tasks(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    symbol_col: str = "symbol",
    label_col: str = "label",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split ``df`` into tasks grouped by ``symbol_col``.

    Each task is represented as a tuple ``(X, y)`` where ``X`` contains the
    selected ``feature_cols`` and ``y`` the labels.
    """
    tasks: List[Tuple[np.ndarray, np.ndarray]] = []
    if symbol_col not in df.columns:
        return tasks
    for _, g in df.groupby(symbol_col):
        X = g[list(feature_cols)].to_numpy(dtype=float)
        y = g[label_col].to_numpy(dtype=float)
        tasks.append((X, y))
    return tasks


# ---------------------------------------------------------------------------
# Meta training helpers
# ---------------------------------------------------------------------------

def train_meta_initialisation(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    symbol_col: str = "symbol",
    label_col: str = "label",
    inner_steps: int = 5,
    inner_lr: float = 0.1,
    meta_lr: float = 0.1,
) -> np.ndarray:
    """Train meta weights from symbol grouped tasks.

    Parameters mirror :class:`ReptileMetaLearner.train`.
    """
    tasks = sample_symbol_tasks(df, feature_cols, symbol_col=symbol_col, label_col=label_col)
    if not tasks:
        raise ValueError("no tasks found for meta learning")
    dim = len(feature_cols)
    meta = ReptileMetaLearner(dim)
    meta.train(tasks, inner_steps=inner_steps, inner_lr=inner_lr, meta_lr=meta_lr)
    return meta.weights.copy()


def save_meta_weights(weights: Sequence[float], model_path: Path | str) -> None:
    """Persist ``weights`` to ``model_path`` under ``meta_weights`` key."""
    path = Path(model_path)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    data["meta_weights"] = [float(w) for w in weights]
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meta-learn shared initialisation")
    parser.add_argument("csv", help="CSV file with training data")
    parser.add_argument("--model", default="model.json", help="Model JSON to update")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument(
        "--features",
        nargs="*",
        help="Optional list of feature columns; defaults to all except symbol/label",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    feat_cols = (
        args.features
        if args.features
        else [c for c in df.columns if c not in {args.symbol_col, args.label_col}]
    )
    weights = train_meta_initialisation(
        df,
        feat_cols,
        symbol_col=args.symbol_col,
        label_col=args.label_col,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
    )
    save_meta_weights(weights, args.model)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
