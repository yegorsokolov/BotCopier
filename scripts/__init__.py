"""Backward compatibility package.

This stub redirects imports from the legacy ``scripts`` namespace to
``botcopier.scripts`` so existing code continues to function after the
modules were moved inside the :mod:`botcopier` package.
"""

import sys
from importlib import import_module

_module = import_module("botcopier.scripts")
# Import top-level scripts living alongside this stub
try:  # pragma: no cover - optional modules
    from . import symbolic_indicators as _symbolic_indicators

    setattr(_module, "symbolic_indicators", _symbolic_indicators)
except Exception:
    pass
try:  # pragma: no cover - optional modules
    from . import promote_strategy as _promote_strategy

    setattr(_module, "promote_strategy", _promote_strategy)
except Exception:
    pass
globals().update(_module.__dict__)
sys.modules[__name__] = _module
