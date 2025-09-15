"""Simple plugin-style registry for classification metrics."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib.metadata import entry_points
from typing import Dict, Sequence

MetricFn = Callable[[object, object, object | None], object]

_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn | None = None):
    """Register ``fn`` under ``name`` in the metric registry.

    The function can be used either directly::

        def my_metric(y_true, probas, profits=None):
            ...

        register_metric("custom", my_metric)

    or as a decorator::

        @register_metric("custom")
        def my_metric(y_true, probas, profits=None):
            ...
    """

    def _register(func: MetricFn) -> MetricFn:
        _REGISTRY[name] = func
        return func

    if fn is None:
        return _register
    return _register(fn)


def load_plugins(names: Iterable[str] | None = None) -> None:
    """Discover and load metric plugins exposed via entry points."""

    try:
        eps = entry_points(group="botcopier.metrics")
    except TypeError:  # pragma: no cover - compatibility with Python <3.10
        eps = entry_points().get("botcopier.metrics", [])

    for ep in eps:
        if names is not None and ep.name not in names:
            continue
        if ep.name in _REGISTRY:
            continue
        try:
            fn = ep.load()
        except Exception:  # pragma: no cover - third party plugin failure
            continue
        register_metric(ep.name, fn)


def get_metrics(selected: Sequence[str] | None = None) -> Dict[str, MetricFn]:
    """Return metric callables filtered by ``selected`` names.

    If ``selected`` is ``None``, all registered metrics are returned.
    Unknown names are ignored.
    """
    if selected is None:
        return dict(_REGISTRY)
    return {name: _REGISTRY[name] for name in selected if name in _REGISTRY}


__all__ = ["register_metric", "load_plugins", "get_metrics", "MetricFn"]
