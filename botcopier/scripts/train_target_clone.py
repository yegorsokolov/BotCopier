#!/usr/bin/env python3
"""CLI entrypoint for training a target clone model.

This wrapper optionally enables distributed execution using Ray when the
``--distributed`` flag is supplied.  Trials and cross-validation folds are
executed in parallel and model artifacts are written to the provided output
directory, which should point to a shared filesystem such as NFS or S3 when
running on a cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import ray  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    _HAS_RAY = False

from botcopier.models.registry import MODEL_REGISTRY
from botcopier.training.pipeline import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train target clone model")
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY.keys()),
        default="logreg",
        help=f"model type to train ({', '.join(MODEL_REGISTRY.keys())})",
    )
    p.add_argument("--tracking-uri", dest="tracking_uri", type=str, default=None)
    p.add_argument("--experiment-name", dest="experiment_name", type=str, default=None)
    p.add_argument(
        "--distributed",
        action="store_true",
        help="enable Ray for distributed trials and folds",
    )
    p.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="classification metric to compute (repeatable)",
    )
    args = p.parse_args()

    if args.distributed:
        if not _HAS_RAY:
            raise RuntimeError("ray is required for distributed execution")
        ray.init()

    train(
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        distributed=args.distributed,
        metrics=args.metrics,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
