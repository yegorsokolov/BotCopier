#!/usr/bin/env python3
"""Select best models and copy to best folder."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

# Support both package and script execution
try:  # pragma: no cover - fallback for script usage
    from .publish_model import publish  # type: ignore
    from .backtest_strategy import run_backtest  # type: ignore
except Exception:  # pragma: no cover
    from publish_model import publish
    from backtest_strategy import run_backtest


def _score_for_model(model_json: Path, metric: str) -> float:
    """Return ``metric`` for ``model_json``.

    If ``evaluation.json`` exists beside ``model_json`` it is preferred.  Otherwise
    metrics are read from ``model.json`` itself.  A special ``metric`` value of
    ``risk_reward`` computes ``expected_return - downside_risk`` from the model
    metadata.
    """

    eval_file = model_json.parent / "evaluation.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                data = json.load(f)
            return float(data.get(metric, 0.0))
        except Exception:
            return 0.0
    try:
        with open(model_json) as f:
            data = json.load(f)
        if metric == "risk_reward":
            er = data.get("expected_return")
            dr = data.get("downside_risk")
            if er is not None and dr is not None:
                return float(er) - float(dr)
            return 0.0
        return float(data.get(metric, 0.0))
    except Exception:
        return 0.0


def _backtest_score(model_json: Path, tick_file: Path, metric: str) -> float:
    """Return backtest ``metric`` for ``model_json`` using ``tick_file``."""

    try:
        result = run_backtest(model_json, tick_file)
        return float(result.get(metric, 0.0))
    except Exception:
        return float("-inf")


def promote(
    models_dir: Path,
    best_dir: Path,
    max_models: int,
    metric: str,
    tick_file: Path,
    backtest_metric: str = "sharpe",
    min_backtest: float = float("-inf"),
    files_dir: Optional[Path] = None,
) -> None:
    """Copy the top ``max_models`` from ``models_dir`` to ``best_dir``.

    Models are first backtested using ``tick_file``.  Their ranking is based on
    the pair ``(metric, backtest_metric)``.  Models failing the
    ``min_backtest`` threshold are skipped.  If ``files_dir`` is provided the
    highest ranked model is also published to that directory so running
    strategies reload the new parameters.
    """

    best_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for m in models_dir.rglob("model_*.json"):
        score = _score_for_model(m, metric)
        bscore = _backtest_score(m, tick_file, backtest_metric)
        if bscore < min_backtest:
            print(
                f"Skipping {m} due to low {backtest_metric}: {bscore}")
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
    p.add_argument('tick_file', help='CSV tick data for backtesting')
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
        Path(args.tick_file),
        args.backtest_metric,
        args.min_backtest,
        Path(args.files_dir) if args.files_dir else None,
    )


if __name__ == '__main__':
    main()
