import pytest

try:
    import numpy  # noqa: F401
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

__all__ = ["HAS_NUMPY"]
