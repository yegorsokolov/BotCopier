"""Benchmarks for the sequence window construction utilities."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

MODULE_NAME = "botcopier.training.sequence_builders"
MODULE_PATH = Path(__file__).resolve().parents[2] / "botcopier" / "training" / "sequence_builders.py"


def _load_sequence_builders_module():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"Unable to import {MODULE_NAME}")

    parent_pkg = "botcopier"
    training_pkg = "botcopier.training"

    if parent_pkg not in sys.modules:
        parent_module = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(parent_pkg, loader=None)
        )
        parent_module.__path__ = [str(MODULE_PATH.parents[1])]
        sys.modules[parent_pkg] = parent_module

    if training_pkg not in sys.modules:
        training_module = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(training_pkg, loader=None)
        )
        training_module.__path__ = [str(MODULE_PATH.parents[0])]
        sys.modules[training_pkg] = training_module

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


sequence_builders = _load_sequence_builders_module()
build_window_sequences = sequence_builders.build_window_sequences

pytest.importorskip("pytest_benchmark")


def _loop_build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    profits: np.ndarray,
    sample_weight: np.ndarray,
    *,
    window_length: int,
):
    indices = range(window_length - 1, X.shape[0])
    seq_list = [X[i - window_length + 1 : i + 1] for i in indices]
    sequence_data = np.stack(seq_list, axis=0).astype(float)
    idx = slice(window_length - 1, None)
    return (
        sequence_data,
        X[idx],
        y[idx],
        profits[idx],
        sample_weight[idx],
    )


def _time_function(func, repeat: int = 3) -> float:
    durations: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return min(durations)


def test_build_window_sequences_benchmark(benchmark):
    rng = np.random.default_rng(1234)
    window_length = 64
    rows, features = 4096, 64
    X = rng.normal(size=(rows, features))
    y = rng.normal(size=rows)
    profits = rng.normal(size=rows)
    sample_weight = rng.random(rows)

    def optimized() -> tuple:
        return build_window_sequences(
            X,
            y,
            profits,
            sample_weight,
            window_length=window_length,
        )

    benchmark.pedantic(optimized, rounds=5, iterations=1)
    optimized_stats = benchmark.stats
    optimized_mean = (
        optimized_stats.stats["mean"]
        if hasattr(optimized_stats, "stats") and isinstance(optimized_stats.stats, dict)
        else optimized_stats.stats.mean  # type: ignore[attr-defined]
    )

    baseline_time = _time_function(
        lambda: _loop_build_sequences(
            X,
            y,
            profits,
            sample_weight,
            window_length=window_length,
        )
    )

    assert optimized_mean * 0.75 < baseline_time
