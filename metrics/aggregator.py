"""Thread-safe metric aggregator keyed by strategy identifier."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import threading

# In-memory storage of metrics per strategy
_METRICS: Dict[str, List[Any]] = defaultdict(list)
_LOCK = threading.Lock()

# Persist metrics alongside this module for easy inspection
_METRICS_DIR = Path(__file__).resolve().parent
_FILE_SUFFIX = ".metrics.json"


def add_metric(strategy_id: str, metric: Any) -> None:
    """Record ``metric`` for ``strategy_id`` and persist to disk."""
    with _LOCK:
        _METRICS[strategy_id].append(metric)
        file_path = _METRICS_DIR / f"{strategy_id}{_FILE_SUFFIX}"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(_METRICS[strategy_id], fh)


def get_aggregated_metrics() -> Dict[str, List[Any]]:
    """Return a copy of all collected metrics grouped by strategy."""
    with _LOCK:
        return {sid: list(values) for sid, values in _METRICS.items()}


def reset_metrics() -> None:
    """Clear all stored metrics and remove persisted files."""
    with _LOCK:
        _METRICS.clear()
        for path in _METRICS_DIR.glob(f"*{_FILE_SUFFIX}"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
