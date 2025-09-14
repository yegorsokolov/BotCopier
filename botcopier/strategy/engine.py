"""Evolutionary/neural search engine assembling DSL expressions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from .dsl import (
    EMA,
    GT,
    LT,
    SMA,
    Add,
    And,
    Constant,
    Div,
    Expr,
    Mul,
    Or,
    Position,
    Price,
    StopLoss,
    Sub,
    backtest,
)


@dataclass
class Candidate:
    """Container for a candidate strategy."""

    expr: Expr
    ret: float
    risk: float
    order_type: str = "market"


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cum = np.cumsum(returns, dtype=float)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd))


def _random_math(
    rng: np.random.Generator, depth: int, sample_window: Callable[[], int]
) -> Expr:
    if depth > 2:
        choice = rng.choice(["price", "const"])
    else:
        choice = rng.choice(
            [
                "price",
                "const",
                "sma",
                "ema",
                "add",
                "sub",
                "mul",
                "div",
            ]
        )
    if choice == "price":
        return Price()
    if choice == "const":
        return Constant(float(rng.uniform(-1, 1)))
    if choice == "sma":
        return SMA(int(sample_window()))
    if choice == "ema":
        return EMA(int(sample_window()))
    left = _random_math(rng, depth + 1, sample_window)
    right = _random_math(rng, depth + 1, sample_window)
    if choice == "add":
        return Add(left, right)
    if choice == "sub":
        return Sub(left, right)
    if choice == "mul":
        return Mul(left, right)
    if choice == "div":
        return Div(left, right)
    raise RuntimeError(f"unknown choice {choice}")


def _random_condition(
    rng: np.random.Generator, sample_window: Callable[[], int]
) -> Expr:
    left = _random_math(rng, 0, sample_window)
    right = _random_math(rng, 0, sample_window)
    return GT(left, right) if rng.random() < 0.5 else LT(left, right)


def _random_strategy(
    rng: np.random.Generator, sample_window: Callable[[], int]
) -> Tuple[Expr, str]:
    cond = _random_condition(rng, sample_window)
    expr: Expr = Position(cond)
    if rng.random() < 0.5:
        expr = StopLoss(expr, float(rng.uniform(0.5, 2.0)))
    order_type = "limit" if rng.random() < 0.5 else "market"
    return expr, order_type


def search_strategies(
    prices: np.ndarray, n_samples: int = 50, seed: int = 0
) -> Tuple[Candidate, List[Candidate]]:
    """Search for profitable strategies and return Pareto front."""

    prices = np.asarray(prices, dtype=float)
    rng = np.random.default_rng(seed)

    if torch is not None:
        rnn = torch.nn.GRU(1, 8, batch_first=True)
        h = torch.zeros(1, 1, 8)
        x = torch.zeros(1, 1, 1)

        def sample_window() -> int:
            nonlocal h, x
            out, h = rnn(x, h)
            return int(torch.sigmoid(out[0, 0, 0]).item() * 18) + 2

    else:  # pragma: no cover - fallback

        def sample_window() -> int:
            return int(rng.integers(2, 20))

    pareto: List[Candidate] = []
    best: Candidate | None = None

    for _ in range(n_samples):
        expr, order_type = _random_strategy(rng, sample_window)
        ret = backtest(prices, expr)
        pnl = np.diff(prices) * expr.eval(prices)[:-1]
        risk = _max_drawdown(pnl)
        cand = Candidate(expr, ret, risk, order_type)

        dominated = False
        for p in pareto:
            if (
                p.ret >= cand.ret
                and p.risk <= cand.risk
                and (p.ret > cand.ret or p.risk < cand.risk)
            ):
                dominated = True
                break
        if dominated:
            continue

        pareto = [
            p
            for p in pareto
            if not (
                cand.ret >= p.ret
                and cand.risk <= p.risk
                and (cand.ret > p.ret or cand.risk < p.risk)
            )
        ]
        pareto.append(cand)
        if best is None or cand.ret > best.ret:
            best = cand

    if best is None:  # pragma: no cover - degenerate fallback
        best = Candidate(Position(GT(Price(), Price())), 0.0, 0.0)
        pareto.append(best)

    return best, pareto


__all__ = ["Candidate", "search_strategies"]
