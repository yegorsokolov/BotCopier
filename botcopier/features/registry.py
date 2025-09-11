"""Simple plugin registry for feature functions."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

try:  # optional pandas/polars types for type checking
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = Any  # type: ignore
try:  # optional polars
    import polars as pl
except Exception:  # pragma: no cover - optional
    pl = Any  # type: ignore

FeatureResult = Tuple[
    Any,
    list[str],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]
FeatureFunc = Callable[..., FeatureResult]

FEATURE_REGISTRY: Dict[str, FeatureFunc] = {}


def register_feature(name: str) -> Callable[[FeatureFunc], FeatureFunc]:
    """Decorator to register a feature function under ``name``."""

    def decorator(func: FeatureFunc) -> FeatureFunc:
        FEATURE_REGISTRY[name] = func
        return func

    return decorator
