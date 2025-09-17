"""Stress testing utilities for trading strategies.

This module provides a light-weight harness that perturbs historical returns
according to a few deterministic stress scenarios and recomputes key risk
metrics.  The scenarios intentionally avoid heavy numerical dependencies so the
logic can be executed in the unit-test environment alongside the promotion
pipeline.  The implementation is therefore based purely on Python standard
library primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class StressResult:
    """Container for stress metrics of a single scenario."""

    pnl: float
    max_drawdown: float
    limit_fill_rate: float | None


def _to_float_list(values: Iterable[float]) -> list[float]:
    """Return ``values`` as a ``list`` of ``float``."""

    return [float(v) for v in values]


def _limit_mask(order_types: Sequence[str] | None, size: int) -> list[bool]:
    """Return a boolean mask of limit orders aligned to the returns array."""

    mask = [False] * size
    if not order_types:
        return mask
    for idx, order in enumerate(order_types):
        if idx >= size:
            break
        if str(order).strip().lower() == "limit":
            mask[idx] = True
    return mask


def _compute_drawdown(returns: Sequence[float]) -> float:
    """Maximum peak-to-trough decline for ``returns``."""

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in returns:
        cumulative += value
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return float(max_dd)


def _compute_fill_rate(limit_mask: Sequence[bool], fill_mask: Sequence[bool]) -> float | None:
    """Fraction of limit orders that remain filled under stress."""

    total_limits = sum(1 for flag in limit_mask if flag)
    if total_limits == 0:
        return None
    filled_limits = sum(
        1 for is_limit, filled in zip(limit_mask, fill_mask) if is_limit and filled
    )
    return filled_limits / total_limits if total_limits else None


def _scenario_metrics(
    returns: Sequence[float],
    limit_mask: Sequence[bool],
    fill_mask: Sequence[bool],
) -> StressResult:
    """Return canonical metrics for a stress scenario."""

    pnl = float(sum(returns)) if returns else 0.0
    drawdown = _compute_drawdown(returns)
    fill_rate = _compute_fill_rate(limit_mask, fill_mask)
    return StressResult(pnl=pnl, max_drawdown=drawdown, limit_fill_rate=fill_rate)


def _shock_returns(returns: Sequence[float]) -> list[float]:
    """Inject a sudden loss into the first observation."""

    stressed = _to_float_list(returns)
    if not stressed:
        return stressed
    mean = sum(stressed) / len(stressed)
    variance = (
        sum((value - mean) ** 2 for value in stressed) / len(stressed)
        if len(stressed) > 1
        else 0.0
    )
    std = math.sqrt(variance)
    baseline = abs(stressed[0])
    magnitude = max(3.0 * std, 0.1 * baseline, 0.01)
    stressed[0] = stressed[0] - magnitude
    return stressed


def _volatility_shift(returns: Sequence[float]) -> list[float]:
    """Increase downside volatility in the latter half of the series."""

    stressed = _to_float_list(returns)
    if len(stressed) <= 1:
        return stressed
    halfway = len(stressed) // 2
    for idx in range(halfway, len(stressed)):
        value = stressed[idx]
        if value >= 0:
            stressed[idx] = value * 0.5
        else:
            stressed[idx] = value * 1.5
    return stressed


def _liquidity_drought(
    returns: Sequence[float], limit_mask: Sequence[bool]
) -> tuple[list[float], list[bool]]:
    """Simulate a liquidity drought by cancelling a fraction of limit orders."""

    stressed = _to_float_list(returns)
    fill_mask = [True] * len(stressed)
    limit_indices = [idx for idx, is_limit in enumerate(limit_mask) if is_limit]
    if not limit_indices:
        return stressed, fill_mask
    # Cancel roughly 30% of limit orders (at least one if present) and halve
    # the remaining ones to mimic partial fills.
    drought = max(1, math.ceil(len(limit_indices) * 0.3))
    cancelled = limit_indices[:drought]
    for idx in cancelled:
        stressed[idx] = 0.0
        fill_mask[idx] = False
    for idx in limit_indices[drought:]:
        stressed[idx] *= 0.5
    return stressed, fill_mask


def run_stress_tests(
    returns: Sequence[float],
    *,
    order_types: Sequence[str] | None = None,
) -> Dict[str, StressResult]:
    """Execute deterministic stress scenarios for ``returns``.

    Parameters
    ----------
    returns:
        Historical per-trade returns of a strategy.  These are assumed to be the
        realised outcomes of the strategy decisions.
    order_types:
        Optional order type annotations aligned with ``returns``.  The liquidity
        stress uses this information to report a limit-order fill rate.

    Returns
    -------
    dict
        Mapping from scenario name to :class:`StressResult`.
    """

    returns_list = _to_float_list(returns)
    limit_mask = _limit_mask(order_types, len(returns_list))
    base_fill_mask = [True] * len(returns_list)

    results: Dict[str, StressResult] = {
        "baseline": _scenario_metrics(returns_list, limit_mask, base_fill_mask),
        "shock": _scenario_metrics(_shock_returns(returns_list), limit_mask, base_fill_mask),
        "volatility_regime": _scenario_metrics(
            _volatility_shift(returns_list), limit_mask, base_fill_mask
        ),
    }
    liquidity_returns, liquidity_fills = _liquidity_drought(returns_list, limit_mask)
    results["liquidity_drought"] = _scenario_metrics(
        liquidity_returns, limit_mask, liquidity_fills
    )
    return results


def summarise_stress_results(results: Mapping[str, StressResult]) -> Dict[str, float | None]:
    """Return aggregate statistics derived from ``results``."""

    pnls = [res.pnl for res in results.values()]
    drawdowns = [res.max_drawdown for res in results.values()]
    fill_rates = [
        res.limit_fill_rate for res in results.values() if res.limit_fill_rate is not None
    ]
    summary: Dict[str, float | None] = {}
    if pnls:
        summary["stress_pnl_min"] = min(pnls)
    if drawdowns:
        summary["stress_drawdown_max"] = max(drawdowns)
    summary["stress_limit_fill_rate_min"] = min(fill_rates) if fill_rates else None
    return summary
