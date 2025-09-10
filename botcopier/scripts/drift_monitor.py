#!/usr/bin/env python3
"""Monitor feature drift using PSI and KS statistics and retrain when necessary."""

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
from sklearn.ensemble import IsolationForest

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


def _ks(expected: pd.Series, actual: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic between two distributions."""
    try:  # pragma: no cover - external dependency
        from scipy.stats import ks_2samp  # type: ignore
    except Exception:  # Fallback implementation
        expected_sorted = np.sort(expected)
        actual_sorted = np.sort(actual)
        all_vals = np.union1d(expected_sorted, actual_sorted)
        cdf1 = np.searchsorted(expected_sorted, all_vals, side="right") / len(expected_sorted)
        cdf2 = np.searchsorted(actual_sorted, all_vals, side="right") / len(actual_sorted)
        return float(np.max(np.abs(cdf1 - cdf2)))
    else:
        return float(ks_2samp(expected, actual).statistic)


def _compute_metrics(baseline_file: Path, recent_file: Path) -> dict[str, float]:
    """Return average PSI and KS across common columns."""
    base = pd.read_csv(baseline_file)
    recent = pd.read_csv(recent_file)
    features = [c for c in base.columns if c in recent.columns]
    psi_vals: list[float] = []
    ks_vals: list[float] = []
    for col in features:
        try:
            b = base[col].astype(float).dropna()
            r = recent[col].astype(float).dropna()
        except Exception:
            continue
        if b.empty or r.empty:
            continue
        psi_vals.append(_psi(b, r))
        ks_vals.append(_ks(b, r))
    return {
        "psi": float(np.mean(psi_vals)) if psi_vals else 0.0,
        "ks": float(np.mean(ks_vals)) if ks_vals else 0.0,
    }


def _isolation_forest_scores(baseline_file: Path, recent_file: Path) -> dict[str, float]:
    """Return per-feature drift scores via :class:`IsolationForest`.

    The baseline distribution is used to fit an ``IsolationForest`` for each
    feature.  The fraction of recent samples predicted as outliers becomes the
    drift score for that feature.  Higher values indicate greater drift.
    """
    base = pd.read_csv(baseline_file)
    recent = pd.read_csv(recent_file)
    features = [c for c in base.columns if c in recent.columns]
    scores: dict[str, float] = {}
    for col in features:
        try:
            b = base[col].astype(float).dropna().to_numpy().reshape(-1, 1)
            r = recent[col].astype(float).dropna().to_numpy().reshape(-1, 1)
        except Exception:
            continue
        if len(b) < 10 or len(r) < 1:
            continue
        model = IsolationForest(random_state=0).fit(b)
        preds = model.predict(r)
        score = float((preds == -1).mean())
        scores[col] = score
    return scores


def _update_model(model_json: Path, metrics: dict[str, float], retrained: bool) -> None:
    """Record drift metrics and retrain timestamp in ``model.json``."""
    ts = datetime.utcnow().isoformat()
    data: dict[str, object] = {}
    if model_json.exists():
        try:
            data = json.loads(model_json.read_text())
        except Exception:
            data = {}
    max_metric = max(metrics.values()) if metrics else 0.0
    data["drift_metric"] = max_metric
    data["drift_metrics"] = metrics
    history = data.setdefault("drift_history", [])
    if isinstance(history, list):
        entry = {"time": ts, **metrics}
        history.append(entry)
    if retrained:
        retrain_hist = data.setdefault("retrain_history", [])
        if isinstance(retrain_hist, list):
            entry = {"time": ts, **metrics}
            retrain_hist.append(entry)
    model_json.write_text(json.dumps(data, indent=2))


def _update_evaluation(eval_file: Path, metrics: dict[str, float]) -> None:
    """Write ``metrics`` to ``evaluation.json`` using ``drift_`` prefixes."""
    data: dict[str, object] = {}
    if eval_file.exists():
        try:
            data = json.loads(eval_file.read_text())
        except Exception:
            data = {}
    data.update({f"drift_{k}": float(v) for k, v in metrics.items()})
    eval_file.write_text(json.dumps(data, indent=2))


def run(
    *,
    baseline_file: Path,
    recent_file: Path,
    drift_threshold: float = 0.2,
    model_json: Path = Path("model.json"),
    log_dir: Path,
    out_dir: Path,
    files_dir: Path,
    drift_scores: Path | None = None,
    flag_file: Path | None = None,
) -> None:
    metrics = _compute_metrics(baseline_file, recent_file)
    logger.info({"drift_metrics": metrics})
    if drift_scores is not None:
        scores = _isolation_forest_scores(baseline_file, recent_file)
        try:
            drift_scores.write_text(json.dumps(scores, indent=2))
        except Exception:
            logger.exception("failed to write drift scores")
        logger.info({"feature_drift": scores})
    method = max(metrics, key=metrics.get)
    metric_val = metrics[method]
    retrain = metric_val > drift_threshold
    _update_model(model_json, metrics, retrain)
    _update_evaluation(model_json.parent / "evaluation.json", metrics)
    if retrain:
        if flag_file is not None:
            try:
                flag_file.parent.mkdir(parents=True, exist_ok=True)
                flag_file.write_text(datetime.utcnow().isoformat())
            except Exception:
                logger.exception("failed to write flag file")
        base = Path(__file__).resolve().parent
        subprocess.run(
            [
                sys.executable,
                str(base / "auto_retrain.py"),
                "--log-dir",
                str(log_dir),
                "--out-dir",
                str(out_dir),
                "--files-dir",
                str(files_dir),
                "--baseline-file",
                str(baseline_file),
                "--recent-file",
                str(recent_file),
                "--drift-method",
                method,
                "--drift-threshold",
                str(drift_threshold),
            ],
            check=True,
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor drift and trigger retraining")
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--recent-file", type=Path, required=True)
    p.add_argument("--drift-threshold", type=float, default=0.2)
    p.add_argument("--model-json", type=Path, default=Path("model.json"))
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--files-dir", type=Path, required=True)
    p.add_argument(
        "--drift-scores",
        type=Path,
        help="optional path to write per-feature IsolationForest drift scores",
    )
    p.add_argument(
        "--flag-file",
        type=Path,
        help="optional file to touch when drift exceeds threshold",
    )
    args = p.parse_args()
    run(
        baseline_file=args.baseline_file,
        recent_file=args.recent_file,
        drift_threshold=args.drift_threshold,
        model_json=args.model_json,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        files_dir=args.files_dir,
        drift_scores=args.drift_scores,
        flag_file=args.flag_file,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
