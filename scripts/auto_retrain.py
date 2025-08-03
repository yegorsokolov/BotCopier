#!/usr/bin/env python3
"""Automatically retrain when metrics fall below thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

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


def _load_column(metrics_file: Path, column: str) -> List[float]:
    values: List[float] = []
    if not metrics_file.exists():
        return values
    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                values.append(float(row.get(column, 0) or 0))
            except Exception:
                continue
    return values


def _compute_psi(base: List[float], new: List[float], bins: int = 10) -> float:
    """Return population stability index between ``base`` and ``new``."""
    if not base or not new:
        return 0.0
    arr_base = np.array(base)
    arr_new = np.array(new)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(arr_base, quantiles))
    if len(edges) <= 1:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    base_hist, _ = np.histogram(arr_base, bins=edges)
    new_hist, _ = np.histogram(arr_new, bins=edges)
    base_perc = base_hist / arr_base.size
    new_perc = new_hist / arr_new.size
    base_perc = np.where(base_perc == 0, 1e-6, base_perc)
    new_perc = np.where(new_perc == 0, 1e-6, new_perc)
    psi = np.sum((base_perc - new_perc) * np.log(base_perc / new_perc))
    return float(psi)


def _compute_ks(base: List[float], new: List[float]) -> float:
    """Return Kolmogorov–Smirnov statistic between ``base`` and ``new``."""
    if not base or not new:
        return 0.0
    arr_base = np.sort(np.array(base))
    arr_new = np.sort(np.array(new))
    all_values = np.sort(np.concatenate([arr_base, arr_new]))
    cdf_base = np.searchsorted(arr_base, all_values, side="right") / arr_base.size
    cdf_new = np.searchsorted(arr_new, all_values, side="right") / arr_new.size
    return float(np.max(np.abs(cdf_base - cdf_new)))


def retrain_if_needed(
    log_dir: Path,
    out_dir: Path,
    files_dir: Path,
    *,
    metrics_file: Optional[Path] = None,
    ref_metrics_file: Optional[Path] = None,
    win_rate_threshold: float = 0.4,
    sharpe_threshold: float = 0.0,
    psi_threshold: Optional[float] = None,
    ks_threshold: Optional[float] = None,
    incremental: bool = True,
) -> bool:
    metrics_path = metrics_file or (log_dir / "metrics.csv")
    row = _load_latest_row(metrics_path)
    if not row:
        return False
    try:
        win_rate = float(row.get("win_rate", 0) or 0)
    except Exception:
        win_rate = 0.0
    try:
        sharpe = float(row.get("sharpe") or row.get("sharpe_ratio") or 0)
    except Exception:
        sharpe = 0.0

    trigger = False
    if win_rate < win_rate_threshold or sharpe < sharpe_threshold:
        trigger = True
    else:
        base_vals: List[float] = []
        new_vals: List[float] = []
        if ref_metrics_file is not None:
            base_vals = _load_column(ref_metrics_file, "win_rate")
            new_vals = _load_column(metrics_path, "win_rate")
        psi = _compute_psi(base_vals, new_vals) if psi_threshold is not None else 0.0
        ks = _compute_ks(base_vals, new_vals) if ks_threshold is not None else 0.0
        if psi_threshold is not None and psi > psi_threshold:
            trigger = True
        if ks_threshold is not None and ks > ks_threshold:
            trigger = True

    if not trigger:
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
        "--training-metrics",
        help="reference metrics CSV for drift detection",
    )
    p.add_argument(
        "--win-rate-threshold",
        type=float,
        default=0.4,
        help="trigger retraining when win rate is below this value",
    )
    p.add_argument(
        "--sharpe-threshold",
        type=float,
        default=0.0,
        help="trigger retraining when Sharpe ratio is below this value",
    )
    p.add_argument(
        "--psi-threshold",
        type=float,
        help="trigger retraining when population stability index exceeds this value",
    )
    p.add_argument(
        "--ks-threshold",
        type=float,
        help="trigger retraining when KS statistic exceeds this value",
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
        ref_metrics_file=Path(args.training_metrics) if args.training_metrics else None,
        win_rate_threshold=args.win_rate_threshold,
        sharpe_threshold=args.sharpe_threshold,
        psi_threshold=args.psi_threshold,
        ks_threshold=args.ks_threshold,
        incremental=not args.no_incremental,
    )


if __name__ == "__main__":
    main()
