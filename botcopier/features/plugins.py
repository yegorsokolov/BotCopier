"""Feature plugin registry with entry point discovery."""
from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, Tuple

FeatureResult = Tuple[
    Any,
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]
FeatureFunc = Callable[..., FeatureResult]

FEATURE_REGISTRY: Dict[str, FeatureFunc] = {}


def register_feature(name: str, fn: FeatureFunc | None = None):
    """Register *fn* under ``name`` in the feature registry.

    Can be used as a decorator when *fn* is ``None``.
    """

    def _register(f: FeatureFunc) -> FeatureFunc:
        FEATURE_REGISTRY[name] = f
        return f

    if fn is None:
        return _register
    return _register(fn)


def load_plugins(names: Iterable[str] | None = None) -> None:
    """Load feature plugins declared via the ``botcopier.features`` entry point.

    Parameters
    ----------
    names:
        Optional iterable of plugin names to load. If omitted, all discovered
        entry points are loaded.
    """
    try:
        eps = entry_points(group="botcopier.features")
    except TypeError:  # pragma: no cover - for older Python
        eps = entry_points().get("botcopier.features", [])

    for ep in eps:
        if names is not None and ep.name not in names:
            continue
        if ep.name in FEATURE_REGISTRY:
            continue
        try:
            fn = ep.load()
        except Exception:
            continue
        register_feature(ep.name, fn)


__all__ = ["FEATURE_REGISTRY", "register_feature", "load_plugins"]
