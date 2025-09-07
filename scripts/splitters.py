"""Custom cross-validation splitters."""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


class PurgedWalkForward:
    """Time series walk-forward splitter with a purging gap.

    Parameters
    ----------
    n_splits:
        Number of folds. Each split uses all data up to a point as the
        training set and the following block as validation.
    gap:
        Number of samples to skip between the end of the training window and
        the start of the validation window. These skipped samples are excluded
        from both sets to avoid look-ahead bias.
    """

    def __init__(self, n_splits: int, gap: int = 1) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if gap < 0:
            raise ValueError("gap must be non-negative")
        self.n_splits = n_splits
        self.gap = gap

    def split(
        self, X: Sequence[object], y: Sequence[object] | None = None, groups: Sequence[object] | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation indices.

        Yields
        ------
        (train_idx, val_idx): tuple of index arrays
            Indices for the training and validation sets where all validation
            indices are strictly greater than the training ones and separated
            by ``gap`` omitted samples.
        """

        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Too few samples for the number of splits")
        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            train_end = test_size * (i + 1)
            val_start = train_end + self.gap
            val_end = val_start + test_size
            if val_start >= n_samples:
                break
            train_idx = indices[:train_end]
            val_idx = indices[val_start:min(n_samples, val_end)]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, X: Sequence[object] | None = None, y: Sequence[object] | None = None, groups: Sequence[object] | None = None) -> int:
        """Return the number of folds."""
        return self.n_splits
