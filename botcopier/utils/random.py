"""Utilities for deterministic behavior across libraries."""
from __future__ import annotations

import random

import numpy as np

try:  # optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda"):
                torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - best effort
            pass
