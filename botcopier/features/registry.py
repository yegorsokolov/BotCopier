"""Runtime feature plugin registry.

This module centralises registration and discovery of feature plugins.  It is
used by :mod:`botcopier.features` as well as third-party packages that expose
additional feature extraction logic via Python entry points.
"""
from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from importlib.metadata import entry_points
from types import ModuleType
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
    """Register ``fn`` under ``name``.

    The returned decorator mirrors :func:`functools.wraps` so plugins can be
    registered either by calling ``register_feature(name, fn)`` directly or by
    using the decorator style ``@register_feature(name)`` above the definition.
    """

    def _register(func: FeatureFunc) -> FeatureFunc:
        FEATURE_REGISTRY[name] = func
        return func

    if fn is None:
        return _register
    return _register(fn)


def _load_internal_plugins() -> None:
    """Import built-in plugin modules lazily."""

    modules: tuple[str, ...] = (
        "botcopier.features.anomaly",
        "botcopier.features.augmentation",
        "botcopier.features.technical",
    )
    for mod_name in modules:
        if mod_name in globals().get("_IMPORTED", set()):
            continue
        try:
            module: ModuleType = import_module(mod_name)
        except Exception:  # pragma: no cover - defensive best effort
            continue
        globals().setdefault("_IMPORTED", set()).add(mod_name)
        # Trigger decorators by touching module attributes.
        getattr(module, "__all__", None)


def load_plugins(names: Iterable[str] | None = None) -> None:
    """Discover feature plugins provided via entry points.

    Parameters
    ----------
    names:
        Optional iterable restricting which plugins should be imported.  When
        omitted, all advertised entry points for the ``botcopier.features``
        group are loaded.
    """

    _load_internal_plugins()

    try:
        eps = entry_points(group="botcopier.features")
    except TypeError:  # pragma: no cover - importlib < 3.10 behaviour
        eps = entry_points().get("botcopier.features", [])

    for ep in eps:
        if names is not None and ep.name not in names:
            continue
        if ep.name in FEATURE_REGISTRY:
            continue
        try:
            func = ep.load()
        except Exception:  # pragma: no cover - ignore faulty plugins
            continue
        register_feature(ep.name, func)


__all__ = ["FEATURE_REGISTRY", "register_feature", "load_plugins"]
