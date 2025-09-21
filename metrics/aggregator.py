"""Thread-safe metric aggregator keyed by strategy identifier."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# In-memory storage of metrics per strategy
_METRICS: Dict[str, List[Any]] = defaultdict(list)
_LOCK = threading.Lock()

# Persist metrics alongside this module for easy inspection
_FILE_SUFFIX = ".metrics.json"
_METRICS_DIR: Path


def _normalise_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _ensure_directory_writable(path: Path) -> Path | None:
    path = path.expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Failed to create metrics directory %s: %s", path, exc)
        return None
    probe = path / "._botcopier_metrics_probe"
    try:
        with probe.open("w", encoding="utf-8") as fh:
            fh.write("")
    except OSError as exc:
        logger.debug("Metrics directory %s is not writable: %s", path, exc)
        return None
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return path


def _default_metrics_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "botcopier-metrics"
    writable = _ensure_directory_writable(base)
    if writable:
        return writable
    fallback = Path(tempfile.mkdtemp(prefix="botcopier-metrics-"))
    writable = _ensure_directory_writable(fallback)
    if writable:
        return writable
    raise RuntimeError("Unable to create a writable metrics directory")


def _strategy_from_path(path: Path) -> str | None:
    name = path.name
    if not name.endswith(_FILE_SUFFIX):
        return None
    return name[: -len(_FILE_SUFFIX)]


def _load_persisted_metrics(directory: Path) -> None:
    for file_path in directory.glob(f"*{_FILE_SUFFIX}"):
        strategy_id = _strategy_from_path(file_path)
        if not strategy_id:
            continue
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to load metrics from %s: %s", file_path, exc)
            continue
        if isinstance(payload, list):
            _METRICS[strategy_id].extend(payload)


def _apply_metrics_dir(directory: Path) -> Path:
    global _METRICS_DIR
    with _LOCK:
        _METRICS.clear()
        _load_persisted_metrics(directory)
        _METRICS_DIR = directory
    return directory


def configure_metrics_dir(metrics_dir: Path | str | None = None) -> Path:
    """Configure the directory used to persist aggregated metrics."""

    candidates: List[Path] = []
    if metrics_dir is not None:
        candidates.append(_normalise_path(metrics_dir))
    env_dir = os.getenv("DATA_METRICS_DIR")
    if env_dir:
        candidates.append(_normalise_path(env_dir))

    resolved: Path | None = None
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        writable = _ensure_directory_writable(candidate)
        if writable is not None:
            resolved = writable
            break
        logger.warning(
            "Metrics directory %s is not writable; falling back to a temporary directory",
            candidate,
        )

    if resolved is None:
        resolved = _default_metrics_dir()
    return _apply_metrics_dir(resolved)


def get_metrics_directory() -> Path:
    """Return the directory currently used for metric persistence."""

    with _LOCK:
        return _METRICS_DIR


# Resolve the metrics directory during import so persisted history is available
_METRICS_DIR = configure_metrics_dir()


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
