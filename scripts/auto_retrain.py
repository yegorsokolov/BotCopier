#!/usr/bin/env python3
"""Automatically retrain when metrics fall below thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Dict

from scripts.train_target_clone import train
from scripts.publish_model import publish


def _load_latest_row(metrics_file: Path) -> Optional[Dict[str, str]]:
    if not metrics_file.exists():
        return None
    last: Optional[Dict[str, str]] = None
    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            last = row
    return last


def retrain_if_needed(
    log_dir: Path,
    out_dir: Path,
    files_dir: Path,
    *,
    metrics_file: Optional[Path] = None,
    win_rate_threshold: float = 0.4,
    incremental: bool = True,
) -> bool:
    metrics_path = metrics_file or (log_dir / "metrics.csv")
    row = _load_latest_row(metrics_path)
    if not row:
        return False
    try:
        win_rate = float(row.get("win_rate", 0) or 0)
    except Exception:
        return False
    if win_rate >= win_rate_threshold:
        return False

    train(log_dir, out_dir, incremental=incremental)
    model_file = out_dir / "model.json"
    publish(model_file, files_dir)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain model when metrics degrade")
    p.add_argument("--log-dir", required=True, help="directory with observer logs")
    p.add_argument("--out-dir", required=True, help="output model directory")
    p.add_argument("--files-dir", required=True, help="MT4 Files directory")
    p.add_argument("--metrics-file", help="path to metrics.csv")
    p.add_argument(
        "--win-rate-threshold",
        type=float,
        default=0.4,
        help="trigger retraining when win rate is below this value",
    )
    p.add_argument(
        "--no-incremental",
        action="store_true",
        help="do not update an existing model incrementally",
    )
    args = p.parse_args()

    retrain_if_needed(
        Path(args.log_dir),
        Path(args.out_dir),
        Path(args.files_dir),
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
        win_rate_threshold=args.win_rate_threshold,
        incremental=not args.no_incremental,
    )


if __name__ == "__main__":
    main()
