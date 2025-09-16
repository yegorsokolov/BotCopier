"""Backward compatibility layer for the strategy search module."""
from __future__ import annotations

from .search import Candidate, search_strategies

__all__ = ["Candidate", "search_strategies"]
