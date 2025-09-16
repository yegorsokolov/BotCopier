"""Prometheus metrics helpers for BotCopier services."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    REGISTRY,
    generate_latest,
    start_http_server,
)

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


def latest_metrics() -> tuple[bytes, str]:
    """Return the current metrics payload and content type.

    This helper is intended for HTTP frameworks that expose metrics via a route
    rather than starting :func:`start_http_server`.  The caller is responsible
    for creating an appropriate :class:`~fastapi.responses.Response` (or
    equivalent) using the returned content.
    """

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
