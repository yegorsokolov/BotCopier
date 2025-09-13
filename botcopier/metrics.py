"""Prometheus metrics helpers for BotCopier services."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram, start_http_server

# Exposed metrics
REQUEST_LATENCY = Histogram(
    "botcopier_latency_seconds",
    "Latency in seconds for operations",
    ["operation"],
)
ERROR_COUNTER = Counter(
    "botcopier_errors_total",
    "Total error count",
    ["type"],
)
TRADE_COUNTER = Counter(
    "botcopier_trades_total",
    "Number of processed trades",
)

# Counter for out-of-distribution samples
OOD_COUNTER = Counter(
    "botcopier_ood_total",
    "Number of OOD samples encountered",
)


def start_metrics_server(port: int) -> None:
    """Start the Prometheus metrics HTTP server on ``port``."""
    start_http_server(port)


@contextmanager
def observe_latency(operation: str) -> Iterator[None]:
    """Context manager recording the latency of ``operation``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        REQUEST_LATENCY.labels(operation=operation).observe(time.perf_counter() - start)
