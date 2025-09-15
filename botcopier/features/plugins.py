"""Backward-compatible shim for :mod:`botcopier.features.registry`."""

from .registry import FEATURE_REGISTRY, load_plugins, register_feature

__all__ = ["FEATURE_REGISTRY", "register_feature", "load_plugins"]
