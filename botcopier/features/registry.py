"""Backward-compatible feature plugin registry.

This module now re-exports symbols from :mod:`botcopier.features.plugins`.
"""
from .plugins import FEATURE_REGISTRY, register_feature, load_plugins

__all__ = ["FEATURE_REGISTRY", "register_feature", "load_plugins"]
