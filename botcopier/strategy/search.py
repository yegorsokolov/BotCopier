"""Neural program search for strategy expressions."""
from __future__ import annotations

import numpy as np

try:  # optional torch dependency
    import torch
except Exception:  # pragma: no cover - fallback
    torch = None  # type: ignore

from .dsl import Price, SMA, GT, Position, backtest, Expr


def _expr_for_window(window: int) -> Expr:
    return Position(GT(Price(), SMA(window)))


def search_strategy(prices: np.ndarray, n_samples: int = 20) -> tuple[Expr, float]:
    """Search for a profitable strategy using an RNN sampler.

    Parameters
    ----------
    prices:
        Price series to evaluate on.
    n_samples:
        Number of candidate programs to sample.
    """
    prices = np.asarray(prices, dtype=float)
    best_expr = _expr_for_window(2)
    best_ret = backtest(prices, best_expr)

    if torch is not None:  # use a tiny RNN to propose windows
        rnn = torch.nn.GRU(1, 8, batch_first=True)
        h = torch.zeros(1, 1, 8)
        x = torch.zeros(1, 1, 1)
        for _ in range(n_samples):
            out, h = rnn(x, h)
            window = int(torch.sigmoid(out[0, 0, 0]).item() * 18) + 2
            expr = _expr_for_window(window)
            ret = backtest(prices, expr)
            if ret > best_ret:
                best_expr, best_ret = expr, ret
    else:  # pragma: no cover - simple random search fallback
        rng = np.random.default_rng(0)
        for _ in range(n_samples):
            window = int(rng.integers(2, 20))
            expr = _expr_for_window(window)
            ret = backtest(prices, expr)
            if ret > best_ret:
                best_expr, best_ret = expr, ret
    return best_expr, best_ret


__all__ = ["search_strategy"]
