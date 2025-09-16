"""Domain specific language primitives for expressible trading strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np


class Expr:
    """Base expression node in the trading DSL."""

    def eval(self, prices: np.ndarray) -> np.ndarray:
        """Evaluate the expression for the provided price series."""

        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the expression."""

        raise NotImplementedError

    def compile(self):
        """Return a callable that evaluates the expression."""

        def _compiled(prices: np.ndarray) -> np.ndarray:
            return self.eval(np.asarray(prices, dtype=float))

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

    def eval(self, prices: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return np.asarray(prices, dtype=float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "price"}


@dataclass
class SMA(Expr):
    """Simple moving average indicator."""

    window: int

    def eval(self, prices: np.ndarray) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        prices = np.asarray(prices, dtype=float)
        if prices.size == 0:
            return np.asarray([], dtype=float)
        window = int(max(1, self.window))
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(prices, kernel, mode="same")

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sma", "window": int(self.window)}


@dataclass
class EMA(Expr):
    """Exponential moving average indicator."""

    window: int

    def eval(self, prices: np.ndarray) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        prices = np.asarray(prices, dtype=float)
        if prices.size == 0:
            return np.asarray([], dtype=float)
        alpha = 2.0 / float(int(self.window) + 1)
        out = np.empty_like(prices, dtype=float)
        out[0] = prices[0]
        for i in range(1, len(prices)):
            out[i] = alpha * prices[i] + (1.0 - alpha) * out[i - 1]
        return out

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "ema", "window": int(self.window)}


@dataclass
class Constant(Expr):
    """Constant numeric value."""

    value: float

    def eval(self, prices: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        prices = np.asarray(prices, dtype=float)
        return np.full(prices.shape, float(self.value), dtype=float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "const", "value": float(self.value)}


@dataclass
class Add(Expr):
    """Addition of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) + self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "add", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Sub(Expr):
    """Subtraction of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) - self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sub", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Mul(Expr):
    """Multiplication of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) * self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "mul", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Div(Expr):
    """Division of two expressions with safe denominator handling."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        denom = self.right.eval(prices)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return self.left.eval(prices) / denom

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "div", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class GT(Expr):
    """Greater-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return (self.left.eval(prices) > self.right.eval(prices)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "gt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class LT(Expr):
    """Less-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return (self.left.eval(prices) < self.right.eval(prices)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "lt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class And(Expr):
    """Logical AND of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return np.logical_and(self.left.eval(prices) > 0, self.right.eval(prices) > 0).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Or(Expr):
    """Logical OR of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return np.logical_or(self.left.eval(prices) > 0, self.right.eval(prices) > 0).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def _children(self) -> Tuple[Expr, ...]:  # pragma: no cover - trivial
        return (self.left, self.right)


@dataclass
class Position(Expr):
    """Position sizing based on a condition."""

    condition: Expr
    size: float = 1.0

    def eval(self, prices: np.ndarray) -> np.ndarray:
        cond = self.condition.eval(prices) > 0
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

    def eval(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.size == 0:
            return prices
        positions = np.asarray(self.child.eval(prices), dtype=float)
        limit = float(abs(self.limit))
        returns = np.diff(prices, prepend=prices[0])
        mask = (returns >= -limit).astype(float)
        return positions * mask

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "stop_loss", "child": self.child.to_dict(), "limit": float(self.limit)}

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
    raise ValueError(f"Unknown expression type {t}")


def backtest(prices: np.ndarray, expr: Expr) -> float:
    """Simple backtest computing the cumulative return of ``expr``."""

    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return 0.0
    positions = np.asarray(expr.eval(prices), dtype=float)
    if positions.size != prices.size:
        raise ValueError("Expression returned array with mismatched shape")
    returns = np.diff(prices)
    pnl = positions[:-1] * returns
    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(pnl))


__all__ = [
    "Expr",
    "Price",
    "SMA",
    "EMA",
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
    "serialize",
    "deserialize",
    "backtest",
]
