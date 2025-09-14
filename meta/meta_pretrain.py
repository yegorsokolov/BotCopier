"""Meta-pretraining utilities for logistic models.

This module exposes helpers to meta-train an initialisation across existing
symbols using either the Reptile or a simple first-order MAML algorithm.  The
resulting weights can be written to ``model.json`` so that downstream training
pipelines can start from a better point and fine-tune on new symbols quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.meta_adapt import ReptileMetaLearner, _logistic_grad

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
    """Split ``df`` into tasks grouped by ``symbol_col``."""

    tasks: List[Tuple[np.ndarray, np.ndarray]] = []
    if symbol_col not in df.columns:
        return tasks
    for _, g in df.groupby(symbol_col):
        X = g[list(feature_cols)].to_numpy(dtype=float)
        y = g[label_col].to_numpy(dtype=float)
        tasks.append((X, y))
    return tasks


# ---------------------------------------------------------------------------
# Meta learning algorithms
# ---------------------------------------------------------------------------


def _maml_train(
    tasks: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    inner_steps: int = 5,
    inner_lr: float = 0.1,
    meta_lr: float = 0.1,
) -> np.ndarray:
    """Train meta weights using a first-order MAML update."""

    dim = tasks[0][0].shape[1]
    w = np.zeros(dim, dtype=float)
    for X, y in tasks:
        w_i = w.copy()
        for _ in range(inner_steps):
            w_i -= inner_lr * _logistic_grad(w_i, X, y)
        grad = _logistic_grad(w_i, X, y)
        w -= meta_lr * grad
    return w


def train_meta_initialisation(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    symbol_col: str = "symbol",
    label_col: str = "label",
    inner_steps: int = 5,
    inner_lr: float = 0.1,
    meta_lr: float = 0.1,
    method: str = "reptile",
) -> np.ndarray:
    """Train meta weights from symbol grouped tasks."""

    tasks = sample_symbol_tasks(
        df, feature_cols, symbol_col=symbol_col, label_col=label_col
    )
    if not tasks:
        raise ValueError("no tasks found for meta learning")
    if method == "maml":
        return _maml_train(
            tasks, inner_steps=inner_steps, inner_lr=inner_lr, meta_lr=meta_lr
        )
    dim = len(feature_cols)
    meta = ReptileMetaLearner(dim)
    meta.train(tasks, inner_steps=inner_steps, inner_lr=inner_lr, meta_lr=meta_lr)
    return meta.weights.copy()


def save_meta_weights(
    weights: Sequence[float],
    model_path: Path | str,
    **meta_info: float | int | str,
) -> None:
    """Persist ``weights`` and metadata to ``model_path``.

    The weights are stored under ``meta['weights']`` along with any provided
    ``meta_info`` (e.g. learning rates or algorithm names).
    """

    path = Path(model_path)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    meta_section = {"weights": [float(w) for w in weights]}
    meta_section.update(meta_info)
    data["meta"] = meta_section
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meta-pretrain shared weights")
    parser.add_argument("csv", help="CSV file with training data")
    parser.add_argument("--model", default="model.json", help="Model JSON to update")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument(
        "--method",
        choices=["reptile", "maml"],
        default="reptile",
        help="Meta learning algorithm to use",
    )
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
        method=args.method,
    )
    save_meta_weights(
        weights,
        args.model,
        method=args.method,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
