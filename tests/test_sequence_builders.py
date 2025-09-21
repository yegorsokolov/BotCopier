"""Regression tests for :mod:`botcopier.training.sequence_builders`."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

MODULE_NAME = "botcopier.training.sequence_builders"
MODULE_PATH = Path(__file__).resolve().parents[1] / "botcopier" / "training" / "sequence_builders.py"

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - defensive guard for CI discovery
    raise RuntimeError(f"Unable to load module specification for {MODULE_NAME}")

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

sequence_builders = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = sequence_builders
spec.loader.exec_module(sequence_builders)


def _loop_build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    profits: np.ndarray,
    sample_weight: np.ndarray,
    *,
    window_length: int,
    returns_df: pd.DataFrame | None = None,
    news_sequences: np.ndarray | None = None,
    symbols: np.ndarray | None = None,
    regime_features: np.ndarray | None = None,
):
    """Mirror the historical Python loop implementation for comparison."""

    indices = range(window_length - 1, X.shape[0])
    seq_list = [X[i - window_length + 1 : i + 1] for i in indices]
    sequence_data = np.stack(seq_list, axis=0).astype(float)

    idx = slice(window_length - 1, None)
    trimmed_returns = (
        returns_df.iloc[idx].reset_index(drop=True) if returns_df is not None else None
    )
    trimmed_news = news_sequences[idx] if news_sequences is not None else None
    trimmed_symbols = symbols[idx] if symbols is not None and symbols.size else symbols
    trimmed_regime = regime_features[idx] if regime_features is not None else None
    return (
        sequence_data,
        trimmed_regime,
        X[idx],
        y[idx],
        profits[idx],
        sample_weight[idx],
        trimmed_returns,
        trimmed_news,
        trimmed_symbols,
    )


def test_build_window_sequences_matches_loop_implementation():
    rng = np.random.default_rng(42)
    rows, features, window_length = 64, 5, 8
    X = rng.normal(size=(rows, features)).astype(np.float32)
    y = rng.normal(size=rows).astype(np.float32)
    profits = rng.normal(size=rows).astype(np.float64)
    sample_weight = rng.random(rows).astype(np.float64)
    returns_df = pd.DataFrame({"ret": rng.normal(size=rows), "vol": rng.normal(size=rows)})
    news_sequences = rng.normal(size=(rows, 3)).astype(np.float32)
    symbols = np.array([f"SYM{i}" for i in range(rows)], dtype=object)
    regime_features = rng.integers(0, 3, size=(rows, 2))

    expected = _loop_build_sequences(
        X,
        y,
        profits,
        sample_weight,
        window_length=window_length,
        returns_df=returns_df,
        news_sequences=news_sequences,
        symbols=symbols,
        regime_features=regime_features,
    )

    result = sequence_builders.build_window_sequences(
        X,
        y,
        profits,
        sample_weight,
        window_length=window_length,
        returns_df=returns_df,
        news_sequences=news_sequences,
        symbols=symbols,
        regime_features=regime_features,
    )

    np.testing.assert_allclose(result[0], expected[0])
    assert result[0].dtype == float
    assert result[0].flags.c_contiguous

    for idx in range(1, 6):
        np.testing.assert_allclose(result[idx], expected[idx])

    if returns_df is not None:
        assert result[6] is not None
        pdt.assert_frame_equal(result[6], expected[6])

    if news_sequences is not None:
        np.testing.assert_allclose(result[7], expected[7])

    if symbols is not None:
        np.testing.assert_array_equal(result[8], expected[8])


def test_build_window_sequences_fallback_matches_loop(monkeypatch):
    rng = np.random.default_rng(7)
    rows, features, window_length = 40, 3, 6
    X = rng.normal(size=(rows, features))
    y = rng.normal(size=rows)
    profits = rng.normal(size=rows)
    sample_weight = rng.random(rows)

    expected = _loop_build_sequences(
        X,
        y,
        profits,
        sample_weight,
        window_length=window_length,
    )

    monkeypatch.setattr(sequence_builders, "_HAS_SLIDING_WINDOW_VIEW", False, raising=False)
    monkeypatch.setattr(sequence_builders, "_sliding_window_view", None, raising=False)

    result = sequence_builders.build_window_sequences(
        X,
        y,
        profits,
        sample_weight,
        window_length=window_length,
    )

    np.testing.assert_allclose(result[0], expected[0])
    assert result[0].flags.c_contiguous
    for res, exp in zip(result[1:6], expected[1:6]):
        if isinstance(exp, np.ndarray):
            np.testing.assert_allclose(res, exp)
        else:
            assert res is exp

    assert result[6] is expected[6]
    assert result[7] is expected[7]
    assert result[8] is expected[8]
