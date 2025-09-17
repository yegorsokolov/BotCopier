"""Domain specific language primitives for expressible trading strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import numpy as np


PriceInput = Union[np.ndarray, Mapping[str, np.ndarray]]
_BASE_SYMBOL_PRIORITY: Tuple[str, ...] = ("base", "target", "self", "price", "close")


def _ensure_array(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr.astype(float, copy=False)


def _sanitize_prices(prices: PriceInput) -> PriceInput:
    if isinstance(prices, Mapping):
        sanitized: Dict[str, np.ndarray] = {}
        changed = False
        for key, value in prices.items():
            if isinstance(value, np.ndarray) and value.ndim == 1 and np.issubdtype(value.dtype, np.floating):
                arr = value.astype(float, copy=False)
            else:
                arr = _ensure_array(value)
            if arr is not value:
                changed = True
            sanitized[key] = arr
        if not changed and isinstance(prices, dict):
            return prices
        return sanitized
    if isinstance(prices, np.ndarray):
        if prices.ndim == 1 and np.issubdtype(prices.dtype, np.floating):
            return prices.astype(float, copy=False)
        return _ensure_array(prices)
    return _ensure_array(prices)


def _infer_base_key(prices: Mapping[str, np.ndarray]) -> str:
    for candidate in _BASE_SYMBOL_PRIORITY:
        if candidate in prices:
            return candidate
    return next(iter(prices))


def _resolve_series(prices: PriceInput, symbol: str | None = None) -> np.ndarray:
    if isinstance(prices, Mapping):
        mapping = prices
        if symbol is None or symbol in ("", "self"):
            key = _infer_base_key(mapping)
            return mapping[key]
        if symbol not in mapping:
            raise KeyError(f"Unknown symbol '{symbol}' in price context")
        return mapping[symbol]
    if symbol not in (None, "", "self"):
        raise KeyError(
            f"Cross-symbol series '{symbol}' requested but only a single price series was provided"
        )
    return prices


def _rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.empty_like(series, dtype=float)
    for idx in range(series.size):
        start = max(0, idx - window + 1)
        out[idx] = float(np.mean(series[start : idx + 1]))
    return out


def _rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.empty_like(series, dtype=float)
    for idx in range(series.size):
        start = max(0, idx - window + 1)
        out[idx] = float(np.std(series[start : idx + 1], ddof=0))
    return out


class Expr:
    """Base expression node in the trading DSL."""

    def eval(self, prices: PriceInput) -> np.ndarray:
        """Evaluate the expression for the provided price series."""

        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the expression."""

        raise NotImplementedError

    def compile(self):
        """Return a callable that evaluates the expression."""

        def _compiled(prices: PriceInput) -> np.ndarray:
            prepared = _sanitize_prices(prices)
            return self.eval(prepared)

        return _compiled

    # The search module relies on the ability to introspect the tree
    # structure.  Implementations override ``_children`` to expose their
    # immediate child expressions.
    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return ()

    def iter_nodes(self) -> Iterable[Expr]:
        """Yield the expression and all sub-expressions."""

        yield self
        for child in self._children():
            yield from child.iter_nodes()


@dataclass
class Price(Expr):
    """Reference to the raw input price series."""

    def eval(self, prices: PriceInput) -> np.ndarray:  # pragma: no cover - trivial
        context = _sanitize_prices(prices)
        return _resolve_series(context)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "price"}


@dataclass
class SMA(Expr):
    """Simple moving average indicator."""

    window: int

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(series, kernel, mode="same")

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sma", "window": int(self.window)}


@dataclass
class EMA(Expr):
    """Exponential moving average indicator."""

    window: int

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        alpha = 2.0 / float(int(self.window) + 1)
        out = np.empty_like(series, dtype=float)
        out[0] = series[0]
        for i in range(1, len(series)):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        return out

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "ema", "window": int(self.window)}


@dataclass
class RSI(Expr):
    """Relative strength index indicator."""

    window: int = 14

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        delta = np.diff(series, prepend=series[0])
        gains = np.clip(delta, 0.0, None)
        losses = np.clip(-delta, 0.0, None)
        avg_gain = _rolling_mean(gains, window)
        avg_loss = _rolling_mean(losses, window)
        denom = np.where(avg_loss < 1e-12, 1e-12, avg_loss)
        rs = avg_gain / denom
        return 100.0 - (100.0 / (1.0 + rs))

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "rsi", "window": int(self.window)}


@dataclass
class ATR(Expr):
    """Average true range indicator using closing price differentials."""

    window: int = 14

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        true_range = np.abs(np.diff(series, prepend=series[0]))
        return _rolling_mean(true_range, window)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "atr", "window": int(self.window)}


@dataclass
class BollingerBand(Expr):
    """Bollinger band indicator returning upper/lower/middle bands."""

    window: int
    num_std: float = 2.0
    band: str = "upper"

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        mean = _rolling_mean(series, window)
        std = _rolling_std(series, window)
        band = str(self.band).lower()
        scale = abs(float(self.num_std))
        if band in {"upper", "up"}:
            return mean + scale * std
        if band in {"lower", "lo", "down"}:
            return mean - scale * std
        if band in {"middle", "mid", "mean", "basis"}:
            return mean
        raise ValueError(f"Unknown Bollinger band '{self.band}'")

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "type": "bollinger",
            "window": int(self.window),
            "num_std": float(self.num_std),
            "band": str(self.band),
        }


@dataclass
class RollingVolatility(Expr):
    """Rolling volatility computed from price returns."""

    window: int

    def eval(self, prices: PriceInput) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        context = _sanitize_prices(prices)
        series = _resolve_series(context)
        if series.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        returns = np.diff(series, prepend=series[0])
        return _rolling_std(returns, window)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "rolling_volatility", "window": int(self.window)}


@dataclass
class CrossPrice(Expr):
    """Price series from a different symbol optionally normalised by the base symbol."""

    symbol: str
    normalize: bool = True

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        other = _resolve_series(context, self.symbol)
        base = _resolve_series(context)
        if other.shape != base.shape:
            raise ValueError("Cross symbol price series length must match base series")
        if self.normalize:
            denom = np.where(np.abs(base) < 1e-12, 1e-12, base)
            return other / denom
        return other

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "cross_price", "symbol": str(self.symbol), "normalize": bool(self.normalize)}


@dataclass
class Constant(Expr):
    """Constant numeric value."""

    value: float

    def eval(self, prices: PriceInput) -> np.ndarray:  # pragma: no cover - trivial
        context = _sanitize_prices(prices)
        base = _resolve_series(context)
        return np.full(base.shape, float(self.value), dtype=float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "const", "value": float(self.value)}


@dataclass
class Add(Expr):
    """Addition of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return self.left.eval(context) + self.right.eval(context)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "add", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Sub(Expr):
    """Subtraction of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return self.left.eval(context) - self.right.eval(context)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sub", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Mul(Expr):
    """Multiplication of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return self.left.eval(context) * self.right.eval(context)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "mul", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Div(Expr):
    """Division of two expressions with safe denominator handling."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        denom = self.right.eval(context)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return self.left.eval(context) / denom

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "div", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class GT(Expr):
    """Greater-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return (self.left.eval(context) > self.right.eval(context)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "gt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class LT(Expr):
    """Less-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return (self.left.eval(context) < self.right.eval(context)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "lt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class And(Expr):
    """Logical AND of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return np.logical_and(self.left.eval(context) > 0, self.right.eval(context) > 0).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Or(Expr):
    """Logical OR of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        return np.logical_or(self.left.eval(context) > 0, self.right.eval(context) > 0).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Position(Expr):
    """Position sizing based on a condition."""

    condition: Expr
    size: float = 1.0

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        cond = self.condition.eval(context) > 0
        size = float(np.clip(self.size, -1.0, 1.0))
        return np.where(cond, size, 0.0)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "position", "condition": self.condition.to_dict(), "size": float(self.size)}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.condition,)


@dataclass
class StopLoss(Expr):
    """Risk guard that zeroes positions after large adverse moves."""

    child: Expr
    limit: float

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        base = _resolve_series(context)
        if base.size == 0:
            return base
        positions = np.asarray(self.child.eval(context), dtype=float)
        if positions.shape != base.shape:
            raise ValueError("Stop loss child expression returned mismatched shape")
        limit = float(abs(self.limit))
        returns = np.diff(base, prepend=base[0])
        mask = (returns >= -limit).astype(float)
        return positions * mask

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "stop_loss", "child": self.child.to_dict(), "limit": float(self.limit)}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.child,)


@dataclass
class TrailingStop(Expr):
    """Trailing stop that adapts to recent highs/lows of the base series."""

    child: Expr
    lookback: int = 10
    buffer: float = 0.0

    def eval(self, prices: PriceInput) -> np.ndarray:
        context = _sanitize_prices(prices)
        base = _resolve_series(context)
        if base.size == 0:
            return base
        positions = np.asarray(self.child.eval(context), dtype=float)
        if positions.shape != base.shape:
            raise ValueError("Trailing stop child expression returned mismatched shape")
        lookback = max(1, int(self.lookback))
        buffer = float(abs(self.buffer))
        out = np.empty_like(positions, dtype=float)
        for idx in range(base.size):
            start = max(0, idx - lookback + 1)
            window = base[start : idx + 1]
            high = float(np.max(window))
            low = float(np.min(window))
            pos = positions[idx]
            if pos > 0 and high - base[idx] > buffer:
                out[idx] = 0.0
            elif pos < 0 and base[idx] - low > buffer:
                out[idx] = 0.0
            else:
                out[idx] = pos
        return out

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "type": "trailing_stop",
            "child": self.child.to_dict(),
            "lookback": int(self.lookback),
            "buffer": float(self.buffer),
        }

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.child,)


def serialize(expr: Expr) -> Dict[str, Any]:
    """Serialize ``expr`` into a JSON compatible dictionary."""

    return expr.to_dict()


def deserialize(data: Dict[str, Any]) -> Expr:
    """Deserialize an expression from a dictionary."""

    t = data["type"]
    if t == "price":
        return Price()
    if t == "sma":
        return SMA(window=int(data["window"]))
    if t == "ema":
        return EMA(window=int(data["window"]))
    if t == "rsi":
        return RSI(window=int(data["window"]))
    if t == "atr":
        return ATR(window=int(data["window"]))
    if t == "bollinger":
        return BollingerBand(
            window=int(data["window"]),
            num_std=float(data.get("num_std", 2.0)),
            band=str(data.get("band", "upper")),
        )
    if t == "rolling_volatility":
        return RollingVolatility(window=int(data["window"]))
    if t == "cross_price":
        return CrossPrice(symbol=str(data["symbol"]), normalize=bool(data.get("normalize", True)))
    if t == "const":
        return Constant(float(data["value"]))
    if t == "add":
        return Add(deserialize(data["left"]), deserialize(data["right"]))
    if t == "sub":
        return Sub(deserialize(data["left"]), deserialize(data["right"]))
    if t == "mul":
        return Mul(deserialize(data["left"]), deserialize(data["right"]))
    if t == "div":
        return Div(deserialize(data["left"]), deserialize(data["right"]))
    if t == "gt":
        return GT(deserialize(data["left"]), deserialize(data["right"]))
    if t == "lt":
        return LT(deserialize(data["left"]), deserialize(data["right"]))
    if t == "and":
        return And(deserialize(data["left"]), deserialize(data["right"]))
    if t == "or":
        return Or(deserialize(data["left"]), deserialize(data["right"]))
    if t == "position":
        return Position(deserialize(data["condition"]), float(data.get("size", 1.0)))
    if t == "stop_loss":
        return StopLoss(deserialize(data["child"]), float(data["limit"]))
    if t == "trailing_stop":
        return TrailingStop(
            deserialize(data["child"]),
            int(data.get("lookback", 10)),
            float(data.get("buffer", 0.0)),
        )
    raise ValueError(f"Unknown expression type {t}")


def backtest(prices: PriceInput, expr: Expr) -> float:
    """Simple backtest computing the cumulative return of ``expr``."""

    context = _sanitize_prices(prices)
    base = _resolve_series(context)
    if base.size < 2:
        return 0.0
    positions = np.asarray(expr.eval(context), dtype=float)
    if positions.shape != base.shape:
        raise ValueError("Expression returned array with mismatched shape")
    returns = np.diff(base)
    pnl = positions[:-1] * returns
    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(pnl))


__all__ = [
    "Expr",
    "Price",
    "SMA",
    "EMA",
    "RSI",
    "ATR",
    "BollingerBand",
    "RollingVolatility",
    "CrossPrice",
    "Constant",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "GT",
    "LT",
    "And",
    "Or",
    "Position",
    "StopLoss",
    "TrailingStop",
    "serialize",
    "deserialize",
    "backtest",
]
