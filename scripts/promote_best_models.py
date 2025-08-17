#!/usr/bin/env python3
"""Select best models and copy to best folder."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Dict

# Support both package and script execution
try:  # pragma: no cover - fallback for script usage
    from .publish_model import publish  # type: ignore
except Exception:  # pragma: no cover
    from publish_model import publish


def _load_metrics(model_json: Path) -> Dict[str, float]:
    """Read metrics from ``evaluation.json`` next to ``model_json``."""

    eval_file = model_json.parent / "evaluation.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                data = json.load(f)
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except Exception:
            return {}
    return {}


def _score_for_model(model_json: Path, metric: str) -> float:
    """Return ``metric`` for ``model_json``.

    Metrics are primarily read from ``evaluation.json``.  If the requested key is
    missing a fallback to ``model.json`` is attempted.  A special ``metric``
    value of ``risk_reward`` computes ``expected_return - downside_risk`` from
    the model metadata.
    """

    if metric == "risk_reward":
        try:
            with open(model_json) as f:
                data = json.load(f)
            er = data.get("expected_return")
            dr = data.get("downside_risk")
            if er is not None and dr is not None:
                return float(er) - float(dr)
            return 0.0
        except Exception:
            return 0.0

    metrics = _load_metrics(model_json)
    if metric in metrics:
        try:
            return float(metrics[metric])
        except Exception:
            return 0.0

    try:
        with open(model_json) as f:
            data = json.load(f)
        return float(data.get(metric, 0.0))
    except Exception:
        return 0.0


def promote(
    models_dir: Path,
    best_dir: Path,
    max_models: int,
    metric: str,
    backtest_metric: str = "sharpe",
    min_backtest: float = float("-inf"),
    files_dir: Optional[Path] = None,
) -> None:
    """Copy the top ``max_models`` from ``models_dir`` to ``best_dir``.

    ``evaluation.json`` files are expected to reside next to each model.  The
    pair ``(metric, backtest_metric)`` obtained from these reports determines
    the ranking.  Models failing the ``min_backtest`` threshold are skipped.  If
    ``files_dir`` is provided the highest ranked model is also published to that
    directory so running strategies reload the new parameters.
    """

    best_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for m in models_dir.rglob("model_*.json"):
        metrics = _load_metrics(m)
        score = _score_for_model(m, metric)
        bscore = float(metrics.get(backtest_metric, 0.0))
        if bscore < min_backtest:
            print(f"Skipping {m} due to low {backtest_metric}: {bscore}")
            continue
        candidates.append((score, bscore, m))

    if not candidates:
        raise ValueError("no models passed backtest threshold")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for idx, (score, bscore, m) in enumerate(candidates[:max_models]):
        dest = best_dir / m.name
        shutil.copy(m, dest)
        print(
            f"Promoted {m} (metric={score}, {backtest_metric}={bscore}) to {dest}")
        if idx == 0 and files_dir is not None:
            publish(dest, files_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models_dir')
    p.add_argument('best_dir')
    p.add_argument('--max-models', type=int, default=3)
    p.add_argument('--metric', default='success_pct',
                   help='metric key to sort by')
    p.add_argument('--backtest-metric', default='sharpe',
                   help='backtest metric key to sort by')
    p.add_argument('--min-backtest', type=float, default=float('-inf'),
                   help='minimum backtest metric to allow promotion')
    p.add_argument('--files-dir', help='MT4 Files directory to publish model')
    args = p.parse_args()
    promote(
        Path(args.models_dir),
        Path(args.best_dir),
        args.max_models,
        args.metric,
        args.backtest_metric,
        args.min_backtest,
        Path(args.files_dir) if args.files_dir else None,
    )


if __name__ == '__main__':
    main()
