"""High level helper for running program search over the trading DSL."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .dsl import Expr
from .engine import search_strategies


def search_strategy(
    prices: np.ndarray, n_samples: int = 50, seed: int = 0
) -> Tuple[Expr, float, float]:
    """Return the best strategy discovered by :func:`search_strategies`.

    Parameters
    ----------
    prices:
        Historical price series used for evaluation.
    n_samples:
        Number of candidate programs to draw during the search.
    seed:
        Random seed controlling the search process.

    Returns
    -------
    tuple
        ``(expr, ret, risk)`` where ``expr`` is the best expression, ``ret`` its
        cumulative return and ``risk`` the associated maximum drawdown.
    """

    best, _ = search_strategies(prices, n_samples=n_samples, seed=seed)
    return best.expr, best.ret, best.risk


__all__ = ["search_strategy"]
