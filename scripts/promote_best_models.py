#!/usr/bin/env python3
"""Select best models and copy to best folder."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from publish_model import publish


def _score_for_model(model_json: Path, metric: str) -> float:
    """Return metric score for ``model_json``.

    The function looks for ``metric`` in the model file itself and then
    in a couple of common report file names (``metrics.json`` or
    ``<stem>_report.json``).  Missing metrics default to ``0``.
    """

    score = 0.0
    try:
        with open(model_json) as f:
            data = json.load(f)
        if metric in data:
            return float(data[metric])
    except Exception:
        return score

    # Look for side-car report files
    for name in (
        "metrics.json",
        "report.json",
        f"{model_json.stem}_metrics.json",
        f"{model_json.stem}_report.json",
    ):
        report = model_json.with_name(name)
        if not report.exists():
            continue
        try:
            with open(report) as f:
                rdata = json.load(f)
            if metric in rdata:
                return float(rdata[metric])
        except Exception:
            continue

    return score


def promote(
    models_dir: Path,
    best_dir: Path,
    max_models: int,
    metric: str,
    files_dir: Optional[Path] = None,
) -> None:
    """Copy the top ``max_models`` from ``models_dir`` to ``best_dir``.

    If ``files_dir`` is provided the highest ranked model is also published to
    that directory so running strategies reload the new parameters.
    """

    best_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for m in models_dir.glob("model_*.json"):
        score = _score_for_model(m, metric)
        candidates.append((score, m))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for idx, (score, m) in enumerate(candidates[:max_models]):
        dest = best_dir / m.name
        shutil.copy(m, dest)
        print(f"Promoted {m} (metric={score}) to {dest}")
        if idx == 0 and files_dir is not None:
            publish(dest, files_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models_dir')
    p.add_argument('best_dir')
    p.add_argument('--max-models', type=int, default=3)
    p.add_argument('--metric', default='success_pct',
                   help='metric key to sort by')
    p.add_argument('--files-dir', help='MT4 Files directory to publish model')
    args = p.parse_args()
    promote(
        Path(args.models_dir),
        Path(args.best_dir),
        args.max_models,
        args.metric,
        Path(args.files_dir) if args.files_dir else None,
    )


if __name__ == '__main__':
    main()
