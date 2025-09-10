from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str, builder: Callable) -> None:
    """Register a model builder under ``name``.

    Parameters
    ----------
    name:
        Identifier for the model.
    builder:
        Callable that returns a fitted model when invoked.
    """
    _REGISTRY[name] = builder


def get_model(name: str) -> Callable:
    """Retrieve a registered model builder by ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not present in the registry.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:  # pragma: no cover - defensive
        raise KeyError(f"Model '{name}' is not registered") from e
