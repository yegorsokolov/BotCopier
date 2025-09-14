#!/usr/bin/env python
"""Profile the training pipeline with cProfile and visualise via snakeviz."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path

import snakeviz.cli as snakeviz_cli

from botcopier.models.registry import MODEL_REGISTRY
from botcopier.training.pipeline import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile pipeline performance")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY.keys()),
        default="logreg",
        help="model type to train",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open generated profiles in a browser using snakeviz",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = args.out_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    profiler = cProfile.Profile()
    profiler.runcall(
        train,
        args.data_dir,
        args.out_dir,
        model_type=args.model_type,
        profile=True,
    )
    full_profile = profiles_dir / "full_pipeline.prof"
    profiler.dump_stats(str(full_profile))

    if args.open:
        for prof in profiles_dir.glob("*.prof"):
            snakeviz_cli.main([str(prof)])


if __name__ == "__main__":  # pragma: no cover
    main()
