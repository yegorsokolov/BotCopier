"""Program synthesis utilities for discovering trading strategies."""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .dsl import (
    Add,
    And,
    Constant,
    Div,
    EMA,
    Expr,
    GT,
    LT,
    Mul,
    Or,
    Position,
    Price,
    SMA,
    StopLoss,
    Sub,
)


DEFAULT_MAX_DEPTH = 4
DEFAULT_MUTATION_RATE = 0.4
DEFAULT_CROSSOVER_RATE = 0.6
DEFAULT_RANDOM_RATE = 0.15
DEFAULT_COMPLEXITY_PENALTY = 0.02
DEFAULT_RISK_PENALTY = 0.15
PARETO_MAX_SIZE = 25


@dataclass
class Candidate:
    """Container representing an evaluated trading program."""

    expr: Expr
    ret: float
    risk: float
    complexity: int


@dataclass
class _NodeRef:
    node: Expr
    parent: Expr | None
    attr: str | None


def _max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    cumulative = np.cumsum(pnl, dtype=float)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    return float(np.max(dd))


def _expression_complexity(expr: Expr) -> int:
    return sum(1 for _ in expr.iter_nodes())


def _iter_nodes(expr: Expr) -> Iterable[_NodeRef]:
    stack: List[_NodeRef] = [_NodeRef(expr, None, None)]
    while stack:
        ref = stack.pop()
        yield ref
        node = ref.node
        if not is_dataclass(node):
            continue
        for field in fields(node):
            value = getattr(node, field.name)
            if isinstance(value, Expr):
                stack.append(_NodeRef(value, node, field.name))


def _random_indicator(rng: np.random.Generator, max_window: int) -> Expr:
    choice = rng.choice(["price", "sma", "ema", "const"])
    if choice == "price":
        return Price()
    if choice == "const":
        return Constant(float(np.clip(rng.normal(0.0, 1.0), -5.0, 5.0)))
    window = int(rng.integers(2, max(3, max_window + 1)))
    if choice == "sma":
        return SMA(window)
    return EMA(window)


def _random_math_expr(
    rng: np.random.Generator, depth: int, max_depth: int, max_window: int
) -> Expr:
    if depth >= max_depth or rng.random() < 0.3:
        return _random_indicator(rng, max_window)
    op = rng.choice([Add, Sub, Mul, Div])
    left = _random_math_expr(rng, depth + 1, max_depth, max_window)
    right = _random_math_expr(rng, depth + 1, max_depth, max_window)
    return op(left, right)


def _random_condition(
    rng: np.random.Generator, depth: int, max_depth: int, max_window: int
) -> Expr:
    left = _random_math_expr(rng, depth + 1, max_depth, max_window)
    right = _random_math_expr(rng, depth + 1, max_depth, max_window)
    cond: Expr = GT(left, right) if rng.random() < 0.5 else LT(left, right)
    if depth < max_depth - 1 and rng.random() < 0.3:
        other = _random_condition(rng, depth + 1, max_depth, max_window)
        cond = And(cond, other) if rng.random() < 0.5 else Or(cond, other)
    return cond


def _random_strategy(
    rng: np.random.Generator, max_window: int, max_depth: int
) -> Expr:
    condition = _random_condition(rng, 0, max_depth, max_window)
    size = float(rng.uniform(-1.0, 1.0))
    if abs(size) < 0.1:
        size = 1.0 if rng.random() < 0.5 else -1.0
    expr: Expr = Position(condition, size)
    if rng.random() < 0.5:
        limit = float(abs(rng.normal(0.5, 0.4))) + 0.05
        expr = StopLoss(expr, limit)
    return expr


def _category(node: Expr) -> str:
    if isinstance(node, (Position, StopLoss)):
        return "strategy"
    if isinstance(node, (GT, LT, And, Or)):
        return "condition"
    if isinstance(node, (Add, Sub, Mul, Div, Price, SMA, EMA, Constant)):
        return "math"
    return "other"


def _mutate(
    expr: Expr,
    rng: np.random.Generator,
    max_window: int,
    max_depth: int,
) -> Expr:
    expr = copy.deepcopy(expr)
    nodes = list(_iter_nodes(expr))
    if not nodes:
        return expr
    ref = rng.choice(nodes)
    node = ref.node
    if isinstance(node, Constant):
        node.value = float(np.clip(node.value + rng.normal(0.0, 0.5), -5.0, 5.0))
        return expr
    if isinstance(node, SMA):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        return expr
    if isinstance(node, EMA):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        return expr
    if isinstance(node, Position):
        if rng.random() < 0.5:
            node.size = float(np.clip(node.size + rng.normal(0.0, 0.4), -1.0, 1.0))
        else:
            node.condition = _random_condition(rng, 0, max_depth, max_window)
        return expr
    if isinstance(node, StopLoss):
        if rng.random() < 0.5:
            node.limit = float(max(0.01, abs(node.limit + rng.normal(0.0, 0.2))))
        else:
            node.child = _random_strategy(rng, max_window, max_depth)
        return expr
    if isinstance(node, (GT, LT, And, Or)):
        new_cond = _random_condition(rng, 0, max_depth, max_window)
        if ref.parent is None:
            return _random_strategy(rng, max_window, max_depth)
        setattr(ref.parent, ref.attr or "child", new_cond)
        return expr
    if isinstance(node, (Add, Sub, Mul, Div, Price, SMA, EMA, Constant)):
        new_math = _random_math_expr(rng, 0, max_depth, max_window)
        if ref.parent is None:
            return Position(_random_condition(rng, 0, max_depth, max_window), 1.0)
        setattr(ref.parent, ref.attr or "child", new_math)
        return expr
    new_subtree = _random_strategy(rng, max_window, max_depth)
    if ref.parent is None:
        return new_subtree
    setattr(ref.parent, ref.attr or "child", new_subtree)
    return expr


def _crossover(
    first: Expr,
    second: Expr,
    rng: np.random.Generator,
    max_window: int,
    max_depth: int,
) -> Expr:
    parent_a = copy.deepcopy(first)
    nodes_a = list(_iter_nodes(parent_a))
    nodes_b = list(_iter_nodes(second))
    if not nodes_a or not nodes_b:
        return parent_a
    categories_a = {}
    for ref in nodes_a:
        categories_a.setdefault(_category(ref.node), []).append(ref)
    categories_b = {}
    for ref in nodes_b:
        categories_b.setdefault(_category(ref.node), []).append(ref)
    shared = [cat for cat in categories_a if cat in categories_b and categories_a[cat] and categories_b[cat]]
    if not shared:
        return _mutate(parent_a, rng, max_window, max_depth)
    cat = rng.choice(shared)
    ref_a = rng.choice(categories_a[cat])
    ref_b = rng.choice(categories_b[cat])
    replacement = copy.deepcopy(ref_b.node)
    if ref_a.parent is None:
        return replacement
    setattr(ref_a.parent, ref_a.attr or "child", replacement)
    return parent_a


def _evaluate(expr: Expr, prices: np.ndarray) -> Candidate | None:
    try:
        positions = np.asarray(expr.eval(prices), dtype=float)
    except Exception:  # pragma: no cover - defensive
        return None
    if positions.shape != prices.shape:
        return None
    returns = np.diff(prices)
    if returns.size == 0:
        pnl = np.zeros(0, dtype=float)
    else:
        pnl = positions[:-1] * returns
    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
    ret = float(np.sum(pnl))
    risk = _max_drawdown(pnl)
    if not (np.isfinite(ret) and np.isfinite(risk)):
        return None
    return Candidate(copy.deepcopy(expr), ret, risk, _expression_complexity(expr))


def _score(candidate: Candidate, risk_penalty: float, complexity_penalty: float) -> float:
    return candidate.ret - risk_penalty * candidate.risk - complexity_penalty * candidate.complexity


def _update_pareto(pareto: List[Candidate], candidate: Candidate) -> List[Candidate]:
    dominated = False
    for other in pareto:
        if (
            other.ret >= candidate.ret
            and other.risk <= candidate.risk
            and (other.ret > candidate.ret or other.risk < candidate.risk)
        ):
            dominated = True
            break
    if dominated:
        return pareto
    pareto = [
        other
        for other in pareto
        if not (
            candidate.ret >= other.ret
            and candidate.risk <= other.risk
            and (candidate.ret > other.ret or candidate.risk < other.risk)
        )
    ]
    pareto.append(candidate)
    pareto.sort(key=lambda c: (c.ret, -c.risk), reverse=True)
    if len(pareto) > PARETO_MAX_SIZE:
        pareto = pareto[:PARETO_MAX_SIZE]
    return pareto


def _baseline_candidate(prices: np.ndarray) -> Candidate:
    baseline = Position(GT(Price(), Price()))
    evaluated = _evaluate(baseline, prices)
    assert evaluated is not None  # pragma: no cover - baseline is always valid
    return evaluated


def search_strategies(
    prices: np.ndarray,
    *,
    n_samples: int = 50,
    seed: int = 0,
    population_size: int | None = None,
    n_generations: int | None = None,
    risk_penalty: float = DEFAULT_RISK_PENALTY,
    complexity_penalty: float = DEFAULT_COMPLEXITY_PENALTY,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    random_rate: float = DEFAULT_RANDOM_RATE,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tuple[Candidate, List[Candidate]]:
    """Run a genetic program search returning the best and Pareto-optimal programs."""

    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        baseline = _baseline_candidate(prices if prices.size else np.asarray([0.0, 0.0]))
        return baseline, [baseline]

    rng = np.random.default_rng(seed)
    if population_size is None:
        population_size = max(8, min(48, n_samples))
    if n_generations is None:
        n_generations = max(5, int(np.ceil(n_samples / max(population_size, 1))))

    max_window = int(np.clip(prices.size // 4, 3, 40))
    population: List[Expr] = [
        _random_strategy(rng, max_window, max_depth) for _ in range(population_size)
    ]

    pareto: List[Candidate] = []
    best: Candidate | None = None
    baseline = _baseline_candidate(prices)
    pareto = _update_pareto(pareto, baseline)

    for _ in range(n_generations):
        evaluated: List[Candidate] = []
        for expr in population:
            cand = _evaluate(expr, prices)
            if cand is None:
                continue
            pareto = _update_pareto(pareto, cand)
            evaluated.append(cand)
        if not evaluated:
            population = [_random_strategy(rng, max_window, max_depth) for _ in range(population_size)]
            continue

        evaluated.sort(key=lambda c: _score(c, risk_penalty, complexity_penalty), reverse=True)
        current_best = evaluated[0]
        if best is None or _score(current_best, risk_penalty, complexity_penalty) > _score(best, risk_penalty, complexity_penalty):
            best = current_best

        elite_count = max(1, int(0.2 * population_size))
        elite = evaluated[:elite_count]

        next_population: List[Expr] = [copy.deepcopy(c.expr) for c in elite]
        while len(next_population) < population_size:
            action = rng.random()
            if action < random_rate:
                child = _random_strategy(rng, max_window, max_depth)
            elif action < random_rate + crossover_rate and len(evaluated) >= 2:
                parent1, parent2 = rng.choice(evaluated, size=2, replace=True)
                child = _crossover(parent1.expr, parent2.expr, rng, max_window, max_depth)
            else:
                parent = rng.choice(evaluated)
                child = copy.deepcopy(parent.expr)
            if rng.random() < mutation_rate:
                child = _mutate(child, rng, max_window, max_depth)
            next_population.append(child)
        population = next_population

    if best is None:
        best = baseline
    pareto = _update_pareto(pareto, best)
    pareto.sort(key=lambda c: (c.ret, -c.risk), reverse=True)
    return best, pareto


def search_strategy(
    prices: np.ndarray, *, n_samples: int = 50, seed: int = 0
) -> Tuple[Expr, float, float]:
    """Convenience helper returning the single best program discovered."""

    best, _ = search_strategies(prices, n_samples=n_samples, seed=seed)
    return best.expr, best.ret, best.risk


__all__ = ["Candidate", "search_strategy", "search_strategies"]
