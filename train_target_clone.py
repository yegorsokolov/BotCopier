"""CLI wrapper for the simplified training pipeline.

This script exists for historical compatibility.  The implementation
moved into the :mod:`botcopier` package and is re-exported here so that
older entry points continue to function.  To preserve the legacy public
API we re-export the training helpers that previously lived directly in
this module.
"""
from __future__ import annotations

import argparse

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features
from botcopier.models.deep import TabTransformer
from botcopier.training.pipeline import detect_resources, run_optuna, train

__all__ = [
    "train",
    "_load_logs",
    "_extract_features",
    "run_optuna",
    "TabTransformer",
    "detect_resources",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-drawdown", type=float, default=None)
    parser.add_argument("--var-limit", type=float, default=None)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    run_optuna(
        n_trials=args.trials,
        max_drawdown=args.max_drawdown,
        var_limit=args.var_limit,
    )
