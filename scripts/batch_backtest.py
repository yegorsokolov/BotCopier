#!/usr/bin/env python3
"""Batch backtest helper.

This small utility runs :mod:`backtest_strategy` across a collection of
parameter files (``model_*.json``) and aggregates the resulting Sharpe ratio,
maximum drawdown and hit rate.  A summary CSV and JSON are written for further
inspection.  After all backtests the results can be fed into
``promote_best_models`` so that only the best performing configurations are
published.

The module exposes :func:`batch_backtest` which is used by the unit tests and a
simple CLI via :func:`main` for manual execution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Optional

# Support both package and script execution
try:  # pragma: no cover - fallback when executed as script
    from .backtest_strategy import run_backtest  # type: ignore
    from .promote_best_models import promote  # type: ignore
except Exception:  # pragma: no cover
    from backtest_strategy import run_backtest  # type: ignore
    from promote_best_models import promote  # type: ignore


def _collect_models(models_dir: Path) -> Iterable[Path]:
    """Yield all parameter files inside ``models_dir``."""

    return sorted(models_dir.rglob("model_*.json"))


def batch_backtest(
    models_dir: Path,
    tick_file: Path,
    summary_csv: Path,
    summary_json: Path,
    best_dir: Optional[Path] = None,
    top_n: int = 3,
    metric: str = "sharpe_ratio",
    files_dir: Optional[Path] = None,
    max_drift: float = 0.2,
) -> List[Dict[str, float]]:
    """Run backtests for all models in ``models_dir``.

    Parameters
    ----------
    models_dir:
        Directory that contains ``model_*.json`` parameter files.
    tick_file:
        CSV with historical market data.
    summary_csv / summary_json:
        Output files for aggregated metrics.
    best_dir:
        If provided, ``promote_best_models.promote`` is invoked to copy the top
        performing models to this directory.
    top_n:
        Number of models to promote.
    metric:
        Metric name used for ranking during promotion.
    files_dir:
        Optional MT4 Files directory to publish models to.
    max_drift:
        Maximum allowed drift metric forwarded to ``promote``.
    """

    results: List[Dict[str, float]] = []

    for model_file in _collect_models(models_dir):
        try:
            metrics = run_backtest(model_file, tick_file)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed backtest for {model_file}: {exc}")
            continue
        result = {
            "model": model_file.name,
            "sharpe": metrics.get("sharpe", 0.0),
            "drawdown": metrics.get("drawdown", 0.0),
            "hit_rate": metrics.get("win_rate", 0.0),
        }
        results.append(result)

    if results:
        fieldnames = ["model", "sharpe", "drawdown", "hit_rate"]
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        with open(summary_json, "w") as f:
            json.dump(results, f, indent=2)

    if best_dir is not None and results:
        promote(
            models_dir,
            best_dir,
            max_models=top_n,
            metric=metric,
            files_dir=files_dir,
            max_drift=max_drift,
        )

    return results


def main() -> None:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Batch backtest multiple models")
    p.add_argument("models_dir", help="Directory containing model_*.json files")
    p.add_argument("tick_file", help="CSV file of tick data")
    p.add_argument("--summary-csv", default="batch_metrics.csv")
    p.add_argument("--summary-json", default="batch_metrics.json")
    p.add_argument("--best-dir", help="Directory to promote best models to")
    p.add_argument("--top-n", type=int, default=3, help="Number of models to promote")
    p.add_argument("--metric", default="sharpe_ratio", help="Metric to rank models by")
    p.add_argument("--files-dir", help="MT4 Files directory to publish model")
    p.add_argument(
        "--max-drift",
        type=float,
        default=0.2,
        help="maximum allowed drift metric before skipping model",
    )
    args = p.parse_args()
    batch_backtest(
        Path(args.models_dir),
        Path(args.tick_file),
        Path(args.summary_csv),
        Path(args.summary_json),
        Path(args.best_dir) if args.best_dir else None,
        args.top_n,
        args.metric,
        Path(args.files_dir) if args.files_dir else None,
        args.max_drift,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
