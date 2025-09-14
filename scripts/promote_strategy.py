"""Evaluate and promote trading strategies.

This script evaluates candidate strategies stored in a *shadow* directory
using out-of-sample returns and promotes strategies that meet risk
requirements.  Evaluation metrics are written to the ``metrics`` directory.
Successful strategies are moved to the ``live`` directory and the registry at
``models/active.json`` is updated.

The evaluation is intentionally simple and does not depend on any third party
libraries to keep tests light-weight.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable


def _load_returns(path: Path) -> Iterable[float]:
    """Load returns from ``path``.

    The file is expected to contain one floating point value per line.  Missing
    files result in an empty list.
    """

    if not path.exists():
        return []
    return [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _max_drawdown(returns: Iterable[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _risk(returns: Iterable[float]) -> float:
    returns = list(returns)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _evaluate(returns: Iterable[float]) -> Dict[str, float]:
    returns = list(returns)
    return {
        "max_drawdown": _max_drawdown(returns),
        "risk": _risk(returns),
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
    }


def promote(
    shadow_dir: Path,
    live_dir: Path,
    metrics_dir: Path,
    registry_path: Path,
    *,
    max_drawdown: float = 0.2,
    max_risk: float = 0.1,
) -> None:
    """Evaluate strategies and promote those passing risk checks."""

    shadow_dir = Path(shadow_dir)
    live_dir = Path(live_dir)
    metrics_dir = Path(metrics_dir)
    registry_path = Path(registry_path)

    live_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry: Dict[str, str] = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except json.JSONDecodeError:
            registry = {}

    for model_dir in shadow_dir.iterdir():
        if not model_dir.is_dir():
            continue
        returns = _load_returns(model_dir / "oos.csv")
        metrics = _evaluate(returns)

        metrics_file = metrics_dir / f"{model_dir.name}.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))

        if metrics["max_drawdown"] <= max_drawdown and metrics["risk"] <= max_risk:
            dest = live_dir / model_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(model_dir), str(dest))
            registry[model_dir.name] = str(dest)

    registry_path.write_text(json.dumps(registry, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Promote strategies that meet risk constraints")
    p.add_argument("--shadow-dir", type=Path, default=Path("shadow"))
    p.add_argument("--live-dir", type=Path, default=Path("live"))
    p.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    p.add_argument("--registry", type=Path, default=Path("models/active.json"))
    p.add_argument("--max-drawdown", type=float, default=0.2)
    p.add_argument("--max-risk", type=float, default=0.1)
    args = p.parse_args()

    promote(
        shadow_dir=args.shadow_dir,
        live_dir=args.live_dir,
        metrics_dir=args.metrics_dir,
        registry_path=args.registry,
        max_drawdown=args.max_drawdown,
        max_risk=args.max_risk,
    )


if __name__ == "__main__":
    main()
