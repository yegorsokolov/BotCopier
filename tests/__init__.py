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

try:
    import sb3_contrib  # noqa: F401
    HAS_SB3_CONTRIB = True
except Exception:
    HAS_SB3_CONTRIB = False

try:
    import tensorflow  # noqa: F401
    HAS_TF = True
except Exception:
    HAS_TF = False

__all__ = ["HAS_NUMPY", "HAS_SB3", "HAS_SB3_CONTRIB", "HAS_TF"]
