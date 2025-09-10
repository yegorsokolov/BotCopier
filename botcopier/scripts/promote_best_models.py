#!/usr/bin/env python3
"""Select best models and copy to best folder."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List

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


def _normalise_metric(metric: str) -> List[str]:
    """Return canonical keys for ``metric``.

    ``evaluate_predictions.py`` writes ``sharpe_ratio`` and ``sortino_ratio``
    while the backtest helper may output ``sharpe``.  Accept a few common
    aliases so users can simply request ``sharpe`` or ``sortino``.
    """

    metric = metric.lower()
    if metric in {"sharpe", "sharpe_ratio"}:
        return ["sharpe_ratio", "sharpe"]
    if metric in {"sortino", "sortino_ratio"}:
        return ["sortino_ratio", "sortino"]
    return [metric]


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
    for key in _normalise_metric(metric):
        if key in metrics:
            try:
                return float(metrics[key])
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
    files_dir: Optional[Path] = None,
    max_drift: float = 0.2,
) -> None:
    """Copy the top ``max_models`` from ``models_dir`` to ``best_dir``.

    Models are ranked purely by the supplied backtest ``metric`` which is read
    from each model's ``evaluation.json`` (falling back to ``model.json`` when
    necessary).  If ``files_dir`` is provided the promoted models are also
    published so running strategies can reload the new parameters.
    """

    best_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for m in models_dir.rglob("model_*.json"):
        metrics = _load_metrics(m)
        if (
            metrics.get("drift_psi", 0.0) > max_drift
            or metrics.get("drift_ks", 0.0) > max_drift
        ):
            continue
        score = _score_for_model(m, metric)
        candidates.append((score, m))

    if not candidates:
        raise ValueError("no models found")

    candidates.sort(key=lambda x: x[0], reverse=True)

    for score, m in candidates[:max_models]:
        dest = best_dir / m.name
        shutil.copy(m, dest)
        print(f"Promoted {m} ({metric}={score}) to {dest}")
        if files_dir is not None:
            publish(dest, files_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models_dir")
    p.add_argument("best_dir")
    p.add_argument("--max-models", type=int, default=3)
    p.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="backtest metric key to sort by",
    )
    p.add_argument("--files-dir", help="MT4 Files directory to publish model")
    p.add_argument(
        "--max-drift",
        type=float,
        default=0.2,
        help="maximum allowed drift metric before skipping model",
    )
    args = p.parse_args()
    promote(
        Path(args.models_dir),
        Path(args.best_dir),
        args.max_models,
        args.metric,
        Path(args.files_dir) if args.files_dir else None,
        args.max_drift,
    )


if __name__ == '__main__':
    main()
