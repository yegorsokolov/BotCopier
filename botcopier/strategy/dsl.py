"""Domain specific language primitives for strategy definition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


class Expr:
    """Base expression node."""

    def eval(self, prices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def compile(self):
        """Return a callable that evaluates the expression."""
        def _compiled(prices: np.ndarray) -> np.ndarray:
            return self.eval(prices)

        return _compiled


@dataclass
class Price(Expr):
    """Reference to raw price series."""

    def eval(self, prices: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return prices

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "price"}


@dataclass
class SMA(Expr):
    """Simple moving average indicator."""

    window: int

    def eval(self, prices: np.ndarray) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        kernel = np.ones(self.window) / float(self.window)
        return np.convolve(prices, kernel, mode="same")

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sma", "window": self.window}


@dataclass
class EMA(Expr):
    """Exponential moving average indicator."""

    window: int

    def eval(self, prices: np.ndarray) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        alpha = 2.0 / float(self.window + 1)
        out = np.empty_like(prices, dtype=float)
        out[0] = prices[0]
        for i in range(1, len(prices)):
            out[i] = alpha * prices[i] + (1.0 - alpha) * out[i - 1]
        return out

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "ema", "window": self.window}


@dataclass
class Constant(Expr):
    """Constant numeric value."""

    value: float

    def eval(self, prices: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return np.full_like(prices, float(self.value))

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "const", "value": self.value}


@dataclass
class Add(Expr):
    """Addition of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) + self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "add", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Sub(Expr):
    """Subtraction of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) - self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "sub", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Mul(Expr):
    """Multiplication of two expressions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return self.left.eval(prices) * self.right.eval(prices)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "mul", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Div(Expr):
    """Division of two expressions with safe zero handling."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        denom = self.right.eval(prices)
        denom = np.where(denom == 0, 1e-12, denom)
        return self.left.eval(prices) / denom

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "div", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class GT(Expr):
    """Greater-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return (self.left.eval(prices) > self.right.eval(prices)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "gt", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class LT(Expr):
    """Less-than logical comparison."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return (self.left.eval(prices) < self.right.eval(prices)).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "lt", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class And(Expr):
    """Logical AND of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return np.logical_and(
            self.left.eval(prices) > 0, self.right.eval(prices) > 0
        ).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Or(Expr):
    """Logical OR of two conditions."""

    left: Expr
    right: Expr

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return np.logical_or(
            self.left.eval(prices) > 0, self.right.eval(prices) > 0
        ).astype(float)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class Position(Expr):
    """Position sizing based on a condition."""

    condition: Expr
    size: float = 1.0

    def eval(self, prices: np.ndarray) -> np.ndarray:
        return np.where(self.condition.eval(prices) > 0, self.size, 0.0)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "type": "position",
            "condition": self.condition.to_dict(),
            "size": self.size,
        }


@dataclass
class StopLoss(Expr):
    """Risk guard that zeroes positions after large adverse moves."""

    child: Expr
    limit: float

    def eval(self, prices: np.ndarray) -> np.ndarray:
        positions = self.child.eval(prices)
        returns = np.diff(prices, prepend=prices[0])
        mask = (returns >= -abs(self.limit)).astype(float)
        return positions * mask

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {
            "type": "stop_loss",
            "child": self.child.to_dict(),
            "limit": self.limit,
        }


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
        return Position(
            deserialize(data["condition"]), float(data.get("size", 1.0))
        )
    if t == "stop_loss":
        return StopLoss(deserialize(data["child"]), float(data["limit"]))
    raise ValueError(f"Unknown expression type {t}")


def backtest(prices: np.ndarray, expr: Expr) -> float:
    """Naive backtest computing cumulative return."""
    prices = np.asarray(prices, dtype=float)
    positions = expr.eval(prices)
    if len(prices) < 2:
        return 0.0
    returns = np.diff(prices)
    pnl = positions[:-1] * returns
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
