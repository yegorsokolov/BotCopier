#!/usr/bin/env python3
"""Sequential drift detection utilities.

This module provides lightweight implementations of the Page-Hinkley and
CUSUM change detection tests.  They operate on a stream of numeric values
representing summary statistics (for example feature means) and emit a
boolean flag when a significant change is detected.  The detectors keep a
small amount of state making them suitable for online processing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    """Classic Page-Hinkley test for detecting distribution shifts.

    Parameters
    ----------
    delta:
        Minimum magnitude of changes to accumulate.  Smaller values make the
        detector more sensitive to noise.
    threshold:
        Detection threshold.  When the accumulated statistic exceeds this
        value a drift event is raised.
    min_samples:
        Number of initial observations required before drift can be reported.
    """

    delta: float = 0.005
    threshold: float = 50.0
    min_samples: int = 30

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self.reset()

    def reset(self) -> None:
        """Reset internal state."""
        self._n = 0
        self._mean = 0.0
        self._cum = 0.0
        self._min_cum = 0.0

    def update(self, value: float) -> bool:
        """Update the detector with ``value``.

        Returns ``True`` when a drift event is detected.
        """

        self._n += 1
        # incremental mean
        self._mean += (value - self._mean) / self._n
        # cumulative difference from the mean, penalised by ``delta``
        self._cum += value - self._mean - self.delta
        self._min_cum = min(self._min_cum, self._cum)
        if self._n >= self.min_samples and self._cum - self._min_cum > self.threshold:
            self.reset()
            return True
        return False


@dataclass
class CusumDetector:
    """Cumulative sum (CUSUM) detector.

    A simple two-sided CUSUM test which tracks positive and negative
    excursions from the running mean.  A drift is reported whenever either
    side exceeds ``threshold``.
    """

    drift: float = 0.0
    threshold: float = 5.0

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._gpos = 0.0
        self._gneg = 0.0

    def update(self, value: float) -> bool:
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._gpos = max(0.0, self._gpos + value - self._mean - self.drift)
        self._gneg = min(0.0, self._gneg + value - self._mean + self.drift)
        if self._gpos > self.threshold or abs(self._gneg) > self.threshold:
            self.reset()
            return True
        return False


__all__ = ["PageHinkley", "CusumDetector"]

