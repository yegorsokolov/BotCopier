#!/usr/bin/env python3
"""Periodically evaluate live predictions against ground truth."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    from otel_logging import setup_logging  # type: ignore
except Exception:  # pragma: no cover
    def setup_logging(name: str) -> None:
        logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)
setup_logging("active_validator")

MetricFn = Callable[[Sequence[int], Sequence[int]], float]


def default_metric(preds: Sequence[int], truth: Sequence[int]) -> float:
    """Simple accuracy metric."""
    if not truth:
        return 0.0
    correct = sum(p == t for p, t in zip(preds, truth))
    return float(correct) / len(truth)


class ActiveValidator:
    """Validate predictions and trigger retraining when performance degrades."""

    def __init__(
        self,
        metric_fn: MetricFn | None = None,
        *,
        threshold: float = 0.8,
        retrain_cb: Callable[[], None] | None = None,
        adjust_cb: Callable[[float], None] | None = None,
        interval: float = 60.0,
    ) -> None:
        self.metric_fn = metric_fn or default_metric
        self.threshold = threshold
        self.retrain_cb = retrain_cb or (lambda: None)
        self.adjust_cb = adjust_cb
        self.interval = interval
        self.history: list[float] = []

    def evaluate(self, preds: Sequence[int], truth: Sequence[int]) -> float:
        """Evaluate one batch of predictions and act on degradation."""
        metric = self.metric_fn(preds, truth)
        self.history.append(metric)
        logger.info("validation metric", extra={"metric": metric})
        if metric < self.threshold:
            logger.warning("metric below threshold", extra={"metric": metric})
            self.retrain_cb()
            if self.adjust_cb:
                self.adjust_cb(metric)
        return metric

    def run(
        self,
        prediction_stream: Iterable[Sequence[int]],
        truth_stream: Iterable[Sequence[int]],
        *,
        iterations: int | None = None,
    ) -> None:
        """Continuously evaluate prediction stream against ground truth."""
        for i, (preds, truth) in enumerate(zip(prediction_stream, truth_stream)):
            self.evaluate(preds, truth)
            if iterations is not None and i + 1 >= iterations:
                break
            time.sleep(self.interval)
