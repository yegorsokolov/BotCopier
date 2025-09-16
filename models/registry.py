from __future__ import annotations

from botcopier.models.registry import MODEL_REGISTRY, get_model, register_model

_REGISTRY = MODEL_REGISTRY

__all__ = ["register_model", "get_model", "_REGISTRY"]
