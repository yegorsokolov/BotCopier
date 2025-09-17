"""Program synthesis utilities for discovering trading strategies."""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .dsl import (
    ATR,
    Add,
    And,
    BollingerBand,
    Constant,
    CrossPrice,
    Div,
    EMA,
    Expr,
    GT,
    LT,
    Mul,
    Or,
    Position,
    Price,
    RSI,
    RollingVolatility,
    SMA,
    StopLoss,
    Sub,
    TrailingStop,
)


DEFAULT_MAX_DEPTH = 4
DEFAULT_MUTATION_RATE = 0.4
DEFAULT_CROSSOVER_RATE = 0.6
DEFAULT_RANDOM_RATE = 0.15
DEFAULT_COMPLEXITY_PENALTY = 0.02
DEFAULT_RISK_PENALTY = 0.15
PARETO_MAX_SIZE = 25
BASE_SYMBOL_PRIORITY: Tuple[str, ...] = ("base", "target", "self", "price", "close")


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
    cost = 0.0
    for node in expr.iter_nodes():
        if isinstance(node, Constant):
            cost += 0.5
        elif isinstance(node, (SMA, EMA, RSI, ATR, BollingerBand, RollingVolatility, CrossPrice)):
            cost += 1.5
        elif isinstance(node, (StopLoss, TrailingStop)):
            cost += 1.0
        else:
            cost += 1.0
    return max(1, int(np.ceil(cost)))


def _risk_adjustment(expr: Expr) -> float:
    has_stop = False
    has_trailing = False
    has_volatility = False
    uses_cross = False
    for node in expr.iter_nodes():
        if isinstance(node, StopLoss):
            has_stop = True
        elif isinstance(node, TrailingStop):
            has_trailing = True
        elif isinstance(node, RollingVolatility):
            has_volatility = True
        elif isinstance(node, CrossPrice):
            uses_cross = True
    factor = 1.0
    if has_stop or has_trailing:
        factor *= 0.8
    if has_stop and has_trailing:
        factor *= 0.85
    if has_volatility:
        factor *= 0.9
    if uses_cross:
        factor *= 1.1
    return max(0.5, min(1.5, factor))


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


def _prepare_prices(
    prices: np.ndarray | Mapping[str, np.ndarray]
) -> Tuple[np.ndarray | Dict[str, np.ndarray], np.ndarray, List[str]]:
    if isinstance(prices, Mapping):
        sanitized: Dict[str, np.ndarray] = {}
        for key, value in prices.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            else:
                arr = arr.astype(float, copy=False)
            sanitized[key] = arr
        if not sanitized:
            base_series = np.asarray([], dtype=float)
            return sanitized, base_series, []
        base_key = next((k for k in BASE_SYMBOL_PRIORITY if k in sanitized), None)
        if base_key is None:
            base_key = next(iter(sanitized))
        base_series = sanitized[base_key]
        cross_symbols = [key for key in sanitized if key != base_key]
        return sanitized, base_series, cross_symbols
    arr = np.asarray(prices, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    else:
        arr = arr.astype(float, copy=False)
    return arr, arr, []


def _random_indicator(
    rng: np.random.Generator, max_window: int, cross_symbols: Sequence[str]
) -> Expr:
    options = [
        "price",
        "sma",
        "ema",
        "rsi",
        "atr",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_middle",
        "volatility",
        "const",
    ]
    if cross_symbols:
        options.append("cross")
    choice = rng.choice(options)
    if choice == "price":
        return Price()
    if choice == "const":
        return Constant(float(np.clip(rng.normal(0.0, 1.0), -5.0, 5.0)))
    window = int(rng.integers(2, max(3, max_window + 1)))
    if choice == "sma":
        return SMA(window)
    if choice == "ema":
        return EMA(window)
    if choice == "rsi":
        return RSI(window)
    if choice == "atr":
        return ATR(window)
    if choice.startswith("bollinger"):
        band = choice.split("_", 1)[-1]
        if band == "middle":
            band = "middle"
        num_std = float(np.clip(rng.normal(2.0, 0.5), 0.5, 4.0))
        return BollingerBand(window=window, num_std=num_std, band=band)
    if choice == "volatility":
        return RollingVolatility(window)
    if choice == "cross":
        symbol = str(rng.choice(cross_symbols))
        normalize = bool(rng.random() < 0.75)
        return CrossPrice(symbol, normalize=normalize)
    return EMA(window)


def _random_math_expr(
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    max_window: int,
    cross_symbols: Sequence[str],
) -> Expr:
    if depth >= max_depth or rng.random() < 0.3:
        return _random_indicator(rng, max_window, cross_symbols)
    op = rng.choice([Add, Sub, Mul, Div])
    left = _random_math_expr(rng, depth + 1, max_depth, max_window, cross_symbols)
    right = _random_math_expr(rng, depth + 1, max_depth, max_window, cross_symbols)
    return op(left, right)


def _random_condition(
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    max_window: int,
    cross_symbols: Sequence[str],
) -> Expr:
    left = _random_math_expr(rng, depth + 1, max_depth, max_window, cross_symbols)
    right = _random_math_expr(rng, depth + 1, max_depth, max_window, cross_symbols)
    cond: Expr = GT(left, right) if rng.random() < 0.5 else LT(left, right)
    if depth < max_depth - 1 and rng.random() < 0.3:
        other = _random_condition(rng, depth + 1, max_depth, max_window, cross_symbols)
        cond = And(cond, other) if rng.random() < 0.5 else Or(cond, other)
    return cond


def _random_strategy(
    rng: np.random.Generator,
    max_window: int,
    max_depth: int,
    cross_symbols: Sequence[str],
) -> Expr:
    condition = _random_condition(rng, 0, max_depth, max_window, cross_symbols)
    size = float(rng.uniform(-1.0, 1.0))
    if abs(size) < 0.1:
        size = 1.0 if rng.random() < 0.5 else -1.0
    expr: Expr = Position(condition, size)
    if rng.random() < 0.5:
        limit = float(abs(rng.normal(0.5, 0.4))) + 0.05
        expr = StopLoss(expr, limit)
    if rng.random() < 0.4:
        lookback = int(np.clip(rng.integers(3, max(4, max_window + 1)), 2, max_window))
        buffer = float(abs(rng.normal(0.2, 0.2)))
        expr = TrailingStop(expr, lookback, buffer)
    return expr


def _category(node: Expr) -> str:
    if isinstance(node, (Position, StopLoss, TrailingStop)):
        return "strategy"
    if isinstance(node, (GT, LT, And, Or)):
        return "condition"
    if isinstance(
        node,
        (
            Add,
            Sub,
            Mul,
            Div,
            Price,
            SMA,
            EMA,
            RSI,
            ATR,
            BollingerBand,
            RollingVolatility,
            CrossPrice,
            Constant,
        ),
    ):
        return "math"
    return "other"


def _mutate(
    expr: Expr,
    rng: np.random.Generator,
    max_window: int,
    max_depth: int,
    cross_symbols: Sequence[str],
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
    if isinstance(node, RSI):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        return expr
    if isinstance(node, ATR):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        return expr
    if isinstance(node, RollingVolatility):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        return expr
    if isinstance(node, BollingerBand):
        node.window = int(np.clip(node.window + rng.integers(-2, 3), 2, max_window))
        node.num_std = float(np.clip(node.num_std + rng.normal(0.0, 0.2), 0.5, 5.0))
        if rng.random() < 0.2:
            node.band = rng.choice(["upper", "lower", "middle"])
        return expr
    if isinstance(node, CrossPrice):
        if cross_symbols and rng.random() < 0.6:
            node.symbol = str(rng.choice(cross_symbols))
        else:
            node.normalize = not node.normalize if rng.random() < 0.5 else node.normalize
        return expr
    if isinstance(node, Position):
        if rng.random() < 0.5:
            node.size = float(np.clip(node.size + rng.normal(0.0, 0.4), -1.0, 1.0))
        else:
            node.condition = _random_condition(rng, 0, max_depth, max_window, cross_symbols)
        return expr
    if isinstance(node, StopLoss):
        if rng.random() < 0.5:
            node.limit = float(max(0.01, abs(node.limit + rng.normal(0.0, 0.2))))
        else:
            node.child = _random_strategy(rng, max_window, max_depth, cross_symbols)
        return expr
    if isinstance(node, TrailingStop):
        if rng.random() < 0.5:
            node.lookback = int(np.clip(node.lookback + rng.integers(-3, 4), 2, max_window))
        elif rng.random() < 0.7:
            node.buffer = float(max(0.0, node.buffer + rng.normal(0.0, 0.1)))
        else:
            node.child = _random_strategy(rng, max_window, max_depth, cross_symbols)
        return expr
    if isinstance(node, (GT, LT, And, Or)):
        new_cond = _random_condition(rng, 0, max_depth, max_window, cross_symbols)
        if ref.parent is None:
            return _random_strategy(rng, max_window, max_depth, cross_symbols)
        setattr(ref.parent, ref.attr or "child", new_cond)
        return expr
    if isinstance(
        node,
        (Add, Sub, Mul, Div, Price, SMA, EMA, RSI, ATR, RollingVolatility, BollingerBand, CrossPrice, Constant),
    ):
        new_math = _random_math_expr(rng, 0, max_depth, max_window, cross_symbols)
        if ref.parent is None:
            return Position(
                _random_condition(rng, 0, max_depth, max_window, cross_symbols), 1.0
            )
        setattr(ref.parent, ref.attr or "child", new_math)
        return expr
    new_subtree = _random_strategy(rng, max_window, max_depth, cross_symbols)
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
    cross_symbols: Sequence[str],
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
        return _mutate(parent_a, rng, max_window, max_depth, cross_symbols)
    cat = rng.choice(shared)
    ref_a = rng.choice(categories_a[cat])
    ref_b = rng.choice(categories_b[cat])
    replacement = copy.deepcopy(ref_b.node)
    if ref_a.parent is None:
        return replacement
    setattr(ref_a.parent, ref_a.attr or "child", replacement)
    return parent_a


def _evaluate(expr: Expr, prices: np.ndarray | Mapping[str, np.ndarray]) -> Candidate | None:
    try:
        base = np.asarray(Price().eval(prices), dtype=float)
        positions = np.asarray(expr.eval(prices), dtype=float)
    except Exception:  # pragma: no cover - defensive
        return None
    if positions.shape != base.shape:
        return None
    returns = np.diff(base)
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
    risk_factor = _risk_adjustment(candidate.expr)
    return candidate.ret - risk_penalty * risk_factor * candidate.risk - complexity_penalty * candidate.complexity


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


def _baseline_candidate(prices: np.ndarray | Mapping[str, np.ndarray]) -> Candidate:
    baseline = Position(GT(Price(), Price()))
    evaluated = _evaluate(baseline, prices)
    assert evaluated is not None  # pragma: no cover - baseline is always valid
    return evaluated


def search_strategies(
    prices: np.ndarray | Mapping[str, np.ndarray],
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

    price_context, base_prices, cross_symbols = _prepare_prices(prices)
    if base_prices.size < 2:
        fallback = base_prices if base_prices.size >= 2 else np.asarray([0.0, 0.0], dtype=float)
        baseline = _baseline_candidate(fallback)
        return baseline, [baseline]

    rng = np.random.default_rng(seed)
    if population_size is None:
        population_size = max(8, min(48, n_samples))
    if n_generations is None:
        n_generations = max(5, int(np.ceil(n_samples / max(population_size, 1))))

    max_window = int(np.clip(base_prices.size // 4, 3, 40))
    population: List[Expr] = [
        _random_strategy(rng, max_window, max_depth, cross_symbols)
        for _ in range(population_size)
    ]

    pareto: List[Candidate] = []
    best: Candidate | None = None
    baseline = _baseline_candidate(price_context)
    pareto = _update_pareto(pareto, baseline)

    for _ in range(n_generations):
        evaluated: List[Candidate] = []
        for expr in population:
            cand = _evaluate(expr, price_context)
            if cand is None:
                continue
            pareto = _update_pareto(pareto, cand)
            evaluated.append(cand)
        if not evaluated:
            population = [
                _random_strategy(rng, max_window, max_depth, cross_symbols)
                for _ in range(population_size)
            ]
            continue

        evaluated.sort(key=lambda c: _score(c, risk_penalty, complexity_penalty), reverse=True)
        current_best = evaluated[0]
        if best is None or _score(current_best, risk_penalty, complexity_penalty) > _score(
            best, risk_penalty, complexity_penalty
        ):
            best = current_best

        elite_count = max(1, int(0.2 * population_size))
        elite = evaluated[:elite_count]

        next_population: List[Expr] = [copy.deepcopy(c.expr) for c in elite]
        while len(next_population) < population_size:
            action = rng.random()
            if action < random_rate:
                child = _random_strategy(rng, max_window, max_depth, cross_symbols)
            elif action < random_rate + crossover_rate and len(evaluated) >= 2:
                parent1, parent2 = rng.choice(evaluated, size=2, replace=True)
                child = _crossover(
                    parent1.expr, parent2.expr, rng, max_window, max_depth, cross_symbols
                )
            else:
                parent = rng.choice(evaluated)
                child = copy.deepcopy(parent.expr)
            if rng.random() < mutation_rate:
                child = _mutate(child, rng, max_window, max_depth, cross_symbols)
            next_population.append(child)
        population = next_population

    if best is None:
        best = baseline
    pareto = _update_pareto(pareto, best)
    pareto.sort(key=lambda c: (c.ret, -c.risk), reverse=True)
    return best, pareto

def search_strategy(
    prices: np.ndarray | Mapping[str, np.ndarray], *, n_samples: int = 50, seed: int = 0
) -> Tuple[Expr, float, float]:
    """Convenience helper returning the single best program discovered."""

    best, _ = search_strategies(prices, n_samples=n_samples, seed=seed)
    return best.expr, best.ret, best.risk


__all__ = ["Candidate", "search_strategy", "search_strategies"]
