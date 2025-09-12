"""Simple metric registry for classification metrics."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

MetricFn = Callable[[object, object, object | None], object]

_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> None:
    """Register a metric callable under ``name``."""
    _REGISTRY[name] = fn


def get_metrics(selected: Sequence[str] | None = None) -> Dict[str, MetricFn]:
    """Return metric callables filtered by ``selected`` names.

    If ``selected`` is ``None``, all registered metrics are returned.
    Unknown names are ignored.
    """
    if selected is None:
        return dict(_REGISTRY)
    return {name: _REGISTRY[name] for name in selected if name in _REGISTRY}
