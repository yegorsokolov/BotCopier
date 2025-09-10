"""Backward compatibility package.

This stub redirects imports from the legacy ``scripts`` namespace to
``botcopier.scripts`` so existing code continues to function after the
modules were moved inside the :mod:`botcopier` package.
"""

from importlib import import_module
import sys

_module = import_module("botcopier.scripts")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
