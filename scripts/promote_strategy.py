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
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from botcopier.scripts.evaluation import evaluate_strategy


logger = logging.getLogger(__name__)


def _load_returns(path: Path) -> Iterable[float]:
    """Load returns from ``path``.

    The file is expected to contain one floating point value per line.  Missing
    files result in an empty list.
    """

    if not path.exists():
        return []
    return [
        float(line.strip()) for line in path.read_text().splitlines() if line.strip()
    ]


def _load_order_types(path: Path) -> Iterable[str]:
    """Load order types from ``path`` if present."""

    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_slippage(path: Path) -> Iterable[float]:
    """Load slippage values from ``path`` if present."""

    if not path.exists():
        return []
    return [
        float(line.strip()) for line in path.read_text().splitlines() if line.strip()
    ]


def promote(
    shadow_dir: Path,
    live_dir: Path,
    metrics_dir: Path,
    registry_path: Path,
    *,
    max_drawdown: float = 0.2,
    max_risk: float = 0.1,
    budget_limit: float = 1.0,
    allowed_order_types: Sequence[str] = ("market", "limit"),
    min_order_compliance: float = 1.0,
    min_stress_pnl: float = 0.0,
    min_liquidity_fill_rate: float | None = None,
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
            logger.exception("Failed to parse registry %s; starting empty", registry_path)
            registry = {}

    risk_report: Dict[str, Dict[str, Any]] = {}

    for model_dir in shadow_dir.iterdir():
        if not model_dir.is_dir():
            continue
        returns = _load_returns(model_dir / "oos.csv")
        order_types = _load_order_types(model_dir / "orders.csv")
        slippage = _load_slippage(model_dir / "slippage.csv")
        metrics = evaluate_strategy(
            returns,
            order_types,
            slippage,
            budget=budget_limit,
            allowed_order_types=allowed_order_types,
        )

        reasons = []
        if metrics["abs_drawdown"] > max_drawdown:
            reasons.append("drawdown")
        if metrics["risk"] > max_risk:
            reasons.append("risk")
        if metrics["budget_utilisation"] > 1.0:
            reasons.append("budget")
        if metrics["order_type_compliance"] < min_order_compliance:
            reasons.append("order_type")
        stress_summary = metrics.get("stress_summary", {})
        stress_pnl_min = stress_summary.get("stress_pnl_min")
        if stress_pnl_min is not None and stress_pnl_min < min_stress_pnl:
            reasons.append("stress_pnl")
        stress_fill = stress_summary.get("stress_limit_fill_rate_min")
        if (
            min_liquidity_fill_rate is not None
            and stress_fill is not None
            and stress_fill < min_liquidity_fill_rate
        ):
            reasons.append("stress_liquidity")

        risk_report[model_dir.name] = {**metrics, "reasons": reasons}

        if not reasons:
            dest = live_dir / model_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(model_dir), str(dest))
            registry[model_dir.name] = str(dest)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    risk_file = metrics_dir / "risk.json"
    risk_file.write_text(json.dumps(risk_report, indent=2))

    registry_path.write_text(json.dumps(registry, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Promote strategies that meet risk constraints"
    )
    p.add_argument("--shadow-dir", type=Path, default=Path("shadow"))
    p.add_argument("--live-dir", type=Path, default=Path("live"))
    p.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    p.add_argument("--registry", type=Path, default=Path("models/active.json"))
    p.add_argument("--max-drawdown", type=float, default=0.2)
    p.add_argument("--max-risk", type=float, default=0.1)
    p.add_argument("--budget-limit", type=float, default=1.0)
    p.add_argument(
        "--allowed-order-types",
        nargs="*",
        default=["market", "limit"],
        help="Permitted order types",
    )
    p.add_argument(
        "--min-order-compliance",
        type=float,
        default=1.0,
        help="Minimum fraction of trades with permitted order types",
    )
    p.add_argument(
        "--min-stress-pnl",
        type=float,
        default=0.0,
        help="Minimum acceptable PnL across stress scenarios",
    )
    p.add_argument(
        "--min-liquidity-fill-rate",
        type=float,
        default=None,
        help="Minimum limit-order fill rate under liquidity stress",
    )
    args = p.parse_args()

    promote(
        shadow_dir=args.shadow_dir,
        live_dir=args.live_dir,
        metrics_dir=args.metrics_dir,
        registry_path=args.registry,
        max_drawdown=args.max_drawdown,
        max_risk=args.max_risk,
        budget_limit=args.budget_limit,
        allowed_order_types=args.allowed_order_types,
        min_order_compliance=args.min_order_compliance,
        min_stress_pnl=args.min_stress_pnl,
        min_liquidity_fill_rate=args.min_liquidity_fill_rate,
    )


if __name__ == "__main__":
    main()
