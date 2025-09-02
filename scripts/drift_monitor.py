#!/usr/bin/env python3
"""Monitor feature drift using PSI or ADWIN and retrain model when necessary."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional dependency
    from river.drift import ADWIN  # type: ignore
except Exception:  # pragma: no cover - optional
    ADWIN = None

try:  # pragma: no cover - fallback for package import
    from otel_logging import setup_logging
except Exception:  # pragma: no cover
    from scripts.otel_logging import setup_logging  # type: ignore

logger = logging.getLogger(__name__)
setup_logging("drift_monitor")


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))
    if len(cut_points) <= 1:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=cut_points)
    actual_counts, _ = np.histogram(actual, bins=cut_points)
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)
    eps = 1e-6
    return float(
        np.sum((expected_perc - actual_perc) * np.log((expected_perc + eps) / (actual_perc + eps)))
    )


def _compute_psi(baseline_file: Path, recent_file: Path) -> float:
    """Average PSI across common columns."""
    base = pd.read_csv(baseline_file)
    recent = pd.read_csv(recent_file)
    features = [c for c in base.columns if c in recent.columns]
    drifts: list[float] = []
    for col in features:
        try:
            b = base[col].astype(float).dropna()
            r = recent[col].astype(float).dropna()
        except Exception:
            continue
        if b.empty or r.empty:
            continue
        drifts.append(_psi(b, r))
    return float(np.mean(drifts)) if drifts else 0.0


def _compute_adwin(baseline_file: Path, recent_file: Path) -> float:
    """Fraction of features where ADWIN detects drift."""
    if ADWIN is None:
        logger.warning("ADWIN not available; returning 0 drift")
        return 0.0
    base = pd.read_csv(baseline_file)
    recent = pd.read_csv(recent_file)
    features = [c for c in base.columns if c in recent.columns]
    if not features:
        return 0.0
    drifted = 0
    for col in features:
        try:
            series = pd.concat([base[col], recent[col]]).astype(float).dropna()
        except Exception:
            continue
        det = ADWIN()
        flagged = False
        for val in series:
            det.update(float(val))
            if det.drift_detected:
                flagged = True
        if flagged:
            drifted += 1
    return drifted / len(features)


def _update_model(model_json: Path, metric: float, method: str, retrained: bool) -> None:
    """Record drift metric and retrain timestamp in ``model.json``."""
    ts = datetime.utcnow().isoformat()
    data: dict[str, object] = {}
    if model_json.exists():
        try:
            data = json.loads(model_json.read_text())
        except Exception:
            data = {}
    data["drift_metric"] = metric
    history = data.setdefault("drift_history", [])
    if isinstance(history, list):
        history.append({"time": ts, "metric": metric, "method": method})
    if retrained:
        retrain_hist = data.setdefault("retrain_history", [])
        if isinstance(retrain_hist, list):
            retrain_hist.append({"time": ts, "metric": metric, "method": method})
    model_json.write_text(json.dumps(data, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor drift and trigger retraining")
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--recent-file", type=Path, required=True)
    p.add_argument("--method", choices=["psi", "adwin"], default="psi")
    p.add_argument("--drift-threshold", type=float, default=0.2)
    p.add_argument("--model-json", type=Path, default=Path("model.json"))
    p.add_argument("--experts-dir", type=Path, default=Path("experts"))
    args = p.parse_args()

    if args.method == "adwin":
        metric = _compute_adwin(args.baseline_file, args.recent_file)
    else:
        metric = _compute_psi(args.baseline_file, args.recent_file)

    logger.info({"drift_metric": metric, "method": args.method})
    retrain = metric > args.drift_threshold
    _update_model(args.model_json, metric, args.method, retrain)
    if retrain:
        base = Path(__file__).resolve().parent
        subprocess.run([sys.executable, str(base / "train_target_clone.py")], check=True)
        subprocess.run(
            [
                sys.executable,
                str(base / "generate_mql4_from_model.py"),
                str(args.model_json),
                str(args.experts_dir),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

