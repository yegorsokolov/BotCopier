import pytest

try:
    import numpy  # noqa: F401
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except Exception:
    HAS_SB3 = False

__all__ = ["HAS_NUMPY", "HAS_SB3"]
