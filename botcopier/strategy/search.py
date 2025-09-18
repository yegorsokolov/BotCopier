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


@dataclass(frozen=True)
class CurriculumStage:
    """Configuration for a curriculum stage during the evolutionary search."""

    name: str
    max_depth: int
    indicator_pool: Tuple[str, ...]
    allow_stop: bool
    allow_trailing: bool
    score_offset: float | None
    min_generations: int
    max_generations: int


@dataclass
class StageProgress:
    """Runtime metadata tracked for each curriculum stage."""

    name: str
    max_depth: int
    threshold: float | None
    score_offset: float | None
    start_generation: int
    end_generation: int | None
    best_score: float
    indicator_pool: Tuple[str, ...]
    allow_stop: bool
    allow_trailing: bool
    min_generations: int
    max_generations: int
    advance_reason: str | None = None

    def to_metadata(self) -> Dict[str, object]:
        """Return a JSON serialisable representation of the progress."""

        best_score = (
            float(self.best_score)
            if np.isfinite(self.best_score)
            else None
        )
        threshold = (
            float(self.threshold)
            if self.threshold is not None
            else None
        )
        return {
            "name": self.name,
            "max_depth": int(self.max_depth),
            "score_threshold": threshold,
            "score_offset": self.score_offset,
            "start_generation": int(self.start_generation),
            "end_generation": (int(self.end_generation)
                                if self.end_generation is not None
                                else None),
            "best_score": best_score,
            "indicator_pool": list(self.indicator_pool),
            "allow_stop": self.allow_stop,
            "allow_trailing": self.allow_trailing,
            "min_generations": int(self.min_generations),
            "max_generations": int(self.max_generations),
            "advance_reason": self.advance_reason,
        }


@dataclass
class SearchOutcome:
    """Container returned by :func:`search_strategies`."""

    best: Candidate
    pareto: List[Candidate]
    metadata: Dict[str, object]

    def __iter__(self):  # pragma: no cover - behaviour verified indirectly
        yield self.best
        yield self.pareto


_SIMPLE_INDICATORS: Tuple[str, ...] = ("price", "sma", "ema", "const")
_MOMENTUM_INDICATORS: Tuple[str, ...] = _SIMPLE_INDICATORS + ("rsi", "atr", "bollinger_middle")
_ADVANCED_INDICATORS: Tuple[str, ...] = _MOMENTUM_INDICATORS + (
    "bollinger_upper",
    "bollinger_lower",
    "volatility",
    "cross",
)


def _build_curriculum(max_depth: int, total_generations: int) -> List[CurriculumStage]:
    """Construct curriculum stages tailored to ``max_depth`` and run length."""

    total_generations = max(1, int(total_generations))
    capped_depth = max(1, int(max_depth))
    stage0_depth = max(1, min(2, capped_depth))
    stage1_depth = max(stage0_depth, min(3, capped_depth))
    stage2_depth = capped_depth

    stage0_min = 1
    stage1_min = 2 if total_generations > 1 else 1
    stage2_min = 2 if total_generations > 2 else 1

    stage0_max = min(total_generations, max(stage0_min + 1, 3))
    stage1_max = min(total_generations, max(stage1_min, 6))
    if stage1_max <= stage0_max:
        stage1_max = min(total_generations, max(stage1_min, stage0_max + 1))
    stage2_max = total_generations

    stages = [
        CurriculumStage(
            name="seed",
            max_depth=stage0_depth,
            indicator_pool=_SIMPLE_INDICATORS,
            allow_stop=False,
            allow_trailing=False,
            score_offset=0.2,
            min_generations=stage0_min,
            max_generations=stage0_max,
        ),
        CurriculumStage(
            name="momentum",
            max_depth=stage1_depth,
            indicator_pool=_MOMENTUM_INDICATORS,
            allow_stop=True,
            allow_trailing=False,
            score_offset=0.45,
            min_generations=stage1_min,
            max_generations=max(stage1_min, stage1_max),
        ),
        CurriculumStage(
            name="advanced",
            max_depth=stage2_depth,
            indicator_pool=_ADVANCED_INDICATORS,
            allow_stop=True,
            allow_trailing=True,
            score_offset=None,
            min_generations=stage2_min,
            max_generations=max(stage2_min, stage2_max),
        ),
    ]

    filtered: List[CurriculumStage] = []
    for stage in stages:
        if filtered and stage.max_depth == filtered[-1].max_depth and stage.indicator_pool == filtered[-1].indicator_pool and stage.allow_stop == filtered[-1].allow_stop and stage.allow_trailing == filtered[-1].allow_trailing:
            # Skip redundant stages if they do not add extra capacity.
            continue
        filtered.append(stage)
    return filtered


class _CurriculumScheduler:
    """Track curriculum stages and decide when to advance."""

    def __init__(self, stages: Sequence[CurriculumStage], baseline_score: float):
        if not stages:
            raise ValueError("curriculum requires at least one stage")
        self._stages = list(stages)
        self._baseline_score = float(baseline_score)
        self._index = 0
        self._progress: List[StageProgress] = []
        self._start_stage(0)

    @property
    def current_stage(self) -> CurriculumStage:
        return self._stages[self._index]

    @property
    def progress(self) -> List[StageProgress]:
        return self._progress

    def update(self, generation: int, best_score: float | None) -> bool:
        progress = self._progress[-1]
        if best_score is not None:
            progress.best_score = max(progress.best_score, best_score)
        progress.end_generation = generation
        if self._index >= len(self._stages) - 1:
            return False
        stage = self.current_stage
        elapsed = generation - progress.start_generation + 1
        threshold = progress.threshold
        should_advance = False
        reason: str | None = None
        if (
            threshold is not None
            and elapsed >= stage.min_generations
            and best_score is not None
            and best_score >= threshold
        ):
            should_advance = True
            reason = "score"
        elif elapsed >= stage.max_generations:
            should_advance = True
            reason = "patience"
        if should_advance:
            progress.advance_reason = reason
            self._advance(generation + 1)
            return True
        return False

    def _start_stage(self, start_generation: int) -> None:
        stage = self._stages[self._index]
        score_offset = stage.score_offset
        threshold = None
        if score_offset is not None and self._index < len(self._stages) - 1:
            threshold = self._baseline_score + score_offset
        self._progress.append(
            StageProgress(
                name=stage.name,
                max_depth=stage.max_depth,
                threshold=threshold,
                score_offset=score_offset,
                start_generation=start_generation,
                end_generation=None,
                best_score=float("-inf"),
                indicator_pool=stage.indicator_pool,
                allow_stop=stage.allow_stop,
                allow_trailing=stage.allow_trailing,
                min_generations=stage.min_generations,
                max_generations=stage.max_generations,
            )
        )

    def _advance(self, next_generation: int) -> None:
        if self._index >= len(self._stages) - 1:
            return
        self._index += 1
        self._start_stage(next_generation)


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
    rng: np.random.Generator,
    max_window: int,
    cross_symbols: Sequence[str],
    indicator_pool: Sequence[str] | None,
) -> Expr:
    if indicator_pool:
        options = [
            option
            for option in indicator_pool
            if option != "cross" or cross_symbols
        ]
        if not options:
            options = ["price"]
    else:
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
    indicator_pool: Sequence[str] | None,
) -> Expr:
    if depth >= max_depth or rng.random() < 0.3:
        return _random_indicator(rng, max_window, cross_symbols, indicator_pool)
    op = rng.choice([Add, Sub, Mul, Div])
    left = _random_math_expr(
        rng, depth + 1, max_depth, max_window, cross_symbols, indicator_pool
    )
    right = _random_math_expr(
        rng, depth + 1, max_depth, max_window, cross_symbols, indicator_pool
    )
    return op(left, right)


def _random_condition(
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    max_window: int,
    cross_symbols: Sequence[str],
    indicator_pool: Sequence[str] | None,
) -> Expr:
    left = _random_math_expr(
        rng, depth + 1, max_depth, max_window, cross_symbols, indicator_pool
    )
    right = _random_math_expr(
        rng, depth + 1, max_depth, max_window, cross_symbols, indicator_pool
    )
    cond: Expr = GT(left, right) if rng.random() < 0.5 else LT(left, right)
    if depth < max_depth - 1 and rng.random() < 0.3:
        other = _random_condition(
            rng, depth + 1, max_depth, max_window, cross_symbols, indicator_pool
        )
        cond = And(cond, other) if rng.random() < 0.5 else Or(cond, other)
    return cond


def _random_strategy(
    rng: np.random.Generator,
    max_window: int,
    max_depth: int,
    cross_symbols: Sequence[str],
    indicator_pool: Sequence[str] | None,
    allow_stop: bool,
    allow_trailing: bool,
) -> Expr:
    condition = _random_condition(
        rng, 0, max_depth, max_window, cross_symbols, indicator_pool
    )
    size = float(rng.uniform(-1.0, 1.0))
    if abs(size) < 0.1:
        size = 1.0 if rng.random() < 0.5 else -1.0
    expr: Expr = Position(condition, size)
    if allow_stop and rng.random() < 0.5:
        limit = float(abs(rng.normal(0.5, 0.4))) + 0.05
        expr = StopLoss(expr, limit)
    if allow_trailing and rng.random() < 0.4:
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
    indicator_pool: Sequence[str] | None,
    allow_stop: bool,
    allow_trailing: bool,
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
            node.condition = _random_condition(
                rng, 0, max_depth, max_window, cross_symbols, indicator_pool
            )
        return expr
    if isinstance(node, StopLoss):
        if rng.random() < 0.5:
            node.limit = float(max(0.01, abs(node.limit + rng.normal(0.0, 0.2))))
        elif allow_stop:
            node.child = _random_strategy(
                rng,
                max_window,
                max_depth,
                cross_symbols,
                indicator_pool,
                allow_stop,
                allow_trailing,
            )
        return expr
    if isinstance(node, TrailingStop):
        if rng.random() < 0.5:
            node.lookback = int(np.clip(node.lookback + rng.integers(-3, 4), 2, max_window))
        elif rng.random() < 0.7:
            node.buffer = float(max(0.0, node.buffer + rng.normal(0.0, 0.1)))
        elif allow_trailing:
            node.child = _random_strategy(
                rng,
                max_window,
                max_depth,
                cross_symbols,
                indicator_pool,
                allow_stop,
                allow_trailing,
            )
        return expr
    if isinstance(node, (GT, LT, And, Or)):
        new_cond = _random_condition(
            rng, 0, max_depth, max_window, cross_symbols, indicator_pool
        )
        if ref.parent is None:
            return _random_strategy(
                rng,
                max_window,
                max_depth,
                cross_symbols,
                indicator_pool,
                allow_stop,
                allow_trailing,
            )
        setattr(ref.parent, ref.attr or "child", new_cond)
        return expr
    if isinstance(
        node,
        (Add, Sub, Mul, Div, Price, SMA, EMA, RSI, ATR, RollingVolatility, BollingerBand, CrossPrice, Constant),
    ):
        new_math = _random_math_expr(
            rng, 0, max_depth, max_window, cross_symbols, indicator_pool
        )
        if ref.parent is None:
            return Position(
                _random_condition(
                    rng, 0, max_depth, max_window, cross_symbols, indicator_pool
                ),
                1.0,
            )
        setattr(ref.parent, ref.attr or "child", new_math)
        return expr
    new_subtree = _random_strategy(
        rng,
        max_window,
        max_depth,
        cross_symbols,
        indicator_pool,
        allow_stop,
        allow_trailing,
    )
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
    indicator_pool: Sequence[str] | None,
    allow_stop: bool,
    allow_trailing: bool,
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
        return _mutate(
            parent_a,
            rng,
            max_window,
            max_depth,
            cross_symbols,
            indicator_pool,
            allow_stop,
            allow_trailing,
        )
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
    use_curriculum: bool = True,
) -> SearchOutcome:
    """Run a genetic program search returning the best program and metadata."""

    price_context, base_prices, cross_symbols = _prepare_prices(prices)
    rng = np.random.default_rng(seed)

    if population_size is None:
        population_size = max(8, min(48, n_samples))
    if n_generations is None:
        n_generations = max(5, int(np.ceil(n_samples / max(population_size, 1))))

    max_window = int(np.clip(base_prices.size // 4, 3, 40))

    if base_prices.size < 2:
        fallback = base_prices if base_prices.size >= 2 else np.asarray([0.0, 0.0], dtype=float)
        baseline = _baseline_candidate(fallback)
        baseline_score = _score(baseline, risk_penalty, complexity_penalty)
        metadata = {
            "use_curriculum": bool(use_curriculum),
            "baseline_score": float(baseline_score),
            "curriculum": [],
            "stage_count": 0,
        }
        return SearchOutcome(baseline, [baseline], metadata)

    baseline = _baseline_candidate(price_context)
    baseline_score = _score(baseline, risk_penalty, complexity_penalty)
    pareto: List[Candidate] = _update_pareto([], baseline)
    best: Candidate | None = None

    scheduler: _CurriculumScheduler | None = None
    static_stage: CurriculumStage | None = None
    if use_curriculum:
        stages = _build_curriculum(max_depth, n_generations)
        scheduler = _CurriculumScheduler(stages, baseline_score)
    else:
        static_stage = CurriculumStage(
            name="static",
            max_depth=max(1, int(max_depth)),
            indicator_pool=_ADVANCED_INDICATORS,
            allow_stop=True,
            allow_trailing=True,
            score_offset=None,
            min_generations=max(1, n_generations),
            max_generations=max(1, n_generations),
        )

    def _stage() -> CurriculumStage:
        if scheduler is not None:
            return scheduler.current_stage
        assert static_stage is not None
        return static_stage

    def _new_strategy() -> Expr:
        stage_cfg = _stage()
        return _random_strategy(
            rng,
            max_window,
            stage_cfg.max_depth,
            cross_symbols,
            stage_cfg.indicator_pool,
            stage_cfg.allow_stop,
            stage_cfg.allow_trailing,
        )

    population: List[Expr] = [_new_strategy() for _ in range(population_size)]

    for generation in range(n_generations):
        stage_cfg = _stage()
        evaluated: List[Candidate] = []
        for expr in population:
            cand = _evaluate(expr, price_context)
            if cand is None:
                continue
            pareto = _update_pareto(pareto, cand)
            evaluated.append(cand)
        if not evaluated:
            if scheduler is not None:
                scheduler.update(generation, None)
                stage_cfg = _stage()
            population = [_new_strategy() for _ in range(population_size)]
            continue

        evaluated.sort(key=lambda c: _score(c, risk_penalty, complexity_penalty), reverse=True)
        current_best = evaluated[0]
        current_best_score = _score(current_best, risk_penalty, complexity_penalty)
        if best is None or current_best_score > _score(best, risk_penalty, complexity_penalty):
            best = current_best

        elite_count = max(1, int(0.2 * population_size))
        elite = evaluated[:elite_count]
        next_population: List[Expr] = [copy.deepcopy(c.expr) for c in elite]

        advanced = False
        if scheduler is not None:
            advanced = scheduler.update(generation, current_best_score)
            stage_cfg = _stage()

        while len(next_population) < population_size:
            action = rng.random()
            if action < random_rate:
                child = _random_strategy(
                    rng,
                    max_window,
                    stage_cfg.max_depth,
                    cross_symbols,
                    stage_cfg.indicator_pool,
                    stage_cfg.allow_stop,
                    stage_cfg.allow_trailing,
                )
            elif action < random_rate + crossover_rate and len(evaluated) >= 2:
                parent1, parent2 = rng.choice(evaluated, size=2, replace=True)
                child = _crossover(
                    parent1.expr,
                    parent2.expr,
                    rng,
                    max_window,
                    stage_cfg.max_depth,
                    cross_symbols,
                    stage_cfg.indicator_pool,
                    stage_cfg.allow_stop,
                    stage_cfg.allow_trailing,
                )
            else:
                parent = rng.choice(evaluated)
                child = copy.deepcopy(parent.expr)
            if rng.random() < mutation_rate:
                child = _mutate(
                    child,
                    rng,
                    max_window,
                    stage_cfg.max_depth,
                    cross_symbols,
                    stage_cfg.indicator_pool,
                    stage_cfg.allow_stop,
                    stage_cfg.allow_trailing,
                )
            next_population.append(child)

        if advanced and scheduler is not None:
            refresh = max(1, population_size // 5)
            stage_cfg = _stage()
            for _ in range(refresh):
                idx = int(rng.integers(0, population_size))
                next_population[idx] = _random_strategy(
                    rng,
                    max_window,
                    stage_cfg.max_depth,
                    cross_symbols,
                    stage_cfg.indicator_pool,
                    stage_cfg.allow_stop,
                    stage_cfg.allow_trailing,
                )

        population = next_population

    if best is None:
        best = baseline
    pareto = _update_pareto(pareto, best)
    pareto.sort(key=lambda c: (c.ret, -c.risk), reverse=True)

    if scheduler is not None:
        curriculum_meta = [entry.to_metadata() for entry in scheduler.progress]
        final_stage = scheduler.current_stage
    else:
        curriculum_meta = []
        final_stage = _stage()

    metadata: Dict[str, object] = {
        "use_curriculum": bool(use_curriculum),
        "baseline_score": float(baseline_score),
        "curriculum": curriculum_meta,
        "stage_count": len(curriculum_meta),
        "final_stage": final_stage.name,
        "final_max_depth": int(final_stage.max_depth),
        "max_window": int(max_window),
    }

    return SearchOutcome(best, pareto, metadata)

def search_strategy(
    prices: np.ndarray | Mapping[str, np.ndarray], *, n_samples: int = 50, seed: int = 0
) -> Tuple[Expr, float, float]:
    """Convenience helper returning the single best program discovered."""

    result = search_strategies(prices, n_samples=n_samples, seed=seed)
    best = result.best
    return best.expr, best.ret, best.risk


__all__ = ["Candidate", "SearchOutcome", "search_strategy", "search_strategies"]
