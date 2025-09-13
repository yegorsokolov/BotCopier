from __future__ import annotations

"""Evaluation hook registry and dispatch utilities.

Hooks are simple callables accepting a mutable context ``dict``. They may
compute metrics or perform side effects and can share information through the
context object. Hooks are executed sequentially in the order provided to
:func:`dispatch_hooks`.
"""

from typing import Callable, Dict, Sequence, Any
import math
import numpy as np

HookFn = Callable[[Dict[str, Any]], None]

# Global hook registry preserving insertion order.
_REGISTRY: Dict[str, HookFn] = {}


def register_hook(name: str) -> Callable[[HookFn], HookFn]:
    """Register ``fn`` under ``name`` in the global hook registry.

    Can be used as a decorator::

        @register_hook("precision")
        def precision(ctx):
            ...
    """

    def decorator(fn: HookFn) -> HookFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_hooks() -> Sequence[str]:
    """Return names of all registered hooks."""

    return list(_REGISTRY.keys())


def dispatch_hooks(names: Sequence[str] | None, ctx: Dict[str, Any]) -> None:
    """Execute hooks listed in ``names`` with shared ``ctx``.

    If ``names`` is ``None`` all registered hooks are executed in registration
    order. Missing names are ignored.
    """

    if names is None:
        names = list(_REGISTRY.keys())
    for name in names:
        fn = _REGISTRY.get(name)
        if fn is None:
            continue
        fn(ctx)


# ---------------------------------------------------------------------------
# Default hooks
# ---------------------------------------------------------------------------


@register_hook("precision")
def _precision_hook(ctx: Dict[str, Any]) -> None:
    """Compute precision metric from true/false positives.

    Expects ``tp`` and ``fp`` in ``ctx`` and stores result in
    ``ctx['stats']['precision']``.
    """

    tp = ctx.get("tp", 0)
    fp = ctx.get("fp", 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    ctx.setdefault("stats", {})["precision"] = float(precision)


@register_hook("sharpe")
def _sharpe_hook(ctx: Dict[str, Any]) -> None:
    """Compute Sharpe and Sortino ratios.

    Expected keys in ``ctx``::

        profits: pd.Series of profit per trade
        net_profits: pd.Series of net profit per trade
        downside: profits[profits < 0]
        downside_net: net_profits[net_profits < 0]
        matches: int number of matched trades
        trade_times: pd.Series of close times

    Results are written to ``ctx['stats']`` under the usual metric keys.
    """

    stats = ctx.setdefault("stats", {})
    profits = ctx.get("profits")
    net_profits = ctx.get("net_profits")
    downside = ctx.get("downside")
    downside_net = ctx.get("downside_net")
    matches = int(ctx.get("matches", 0))
    trade_times = ctx.get("trade_times")

    expected_return = float(stats.get("expected_return", 0.0))
    expected_return_net = float(stats.get("expected_return_net", 0.0))

    sharpe = sortino = 0.0
    sharpe_net = sortino_net = 0.0
    if profits is not None and net_profits is not None and matches > 1:
        mean = expected_return
        variance = float(profits.var(ddof=1))
        std = math.sqrt(variance)
        if std > 0:
            sharpe = mean / std
        if downside is not None and len(downside):
            downside_dev = math.sqrt(float((downside**2).mean()))
            if downside_dev > 0:
                sortino = mean / downside_dev
        mean_net = expected_return_net
        variance_net = float(net_profits.var(ddof=1))
        std_net = math.sqrt(variance_net)
        if std_net > 0:
            sharpe_net = mean_net / std_net
        if downside_net is not None and len(downside_net):
            downside_dev_net = math.sqrt(float((downside_net**2).mean()))
            if downside_dev_net > 0:
                sortino_net = mean_net / downside_dev_net

    annual_sharpe = annual_sortino = 0.0
    annual_sharpe_net = annual_sortino_net = 0.0
    if matches > 1 and trade_times is not None and not trade_times.isna().all():
        start = trade_times.min()
        end = trade_times.max()
        years = (end - start).total_seconds() / (365 * 24 * 3600)
        if years <= 0:
            years = 1.0
        trades_per_year = matches / years
        factor = math.sqrt(trades_per_year)
        annual_sharpe = sharpe * factor
        annual_sortino = sortino * factor
        annual_sharpe_net = sharpe_net * factor
        annual_sortino_net = sortino_net * factor

    stats.update(
        {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "sharpe_ratio_net": float(sharpe_net),
            "sortino_ratio_net": float(sortino_net),
            "sharpe_ratio_annualised": float(annual_sharpe),
            "sortino_ratio_annualised": float(annual_sortino),
            "sharpe_ratio_net_annualised": float(annual_sharpe_net),
            "sortino_ratio_net_annualised": float(annual_sortino_net),
        }
    )
