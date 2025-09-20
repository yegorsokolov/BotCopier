from __future__ import annotations

import time

import pytest

np = pytest.importorskip("numpy")

from botcopier.utils.inference import FeaturePipeline


def _make_pipeline() -> FeaturePipeline:
    weights = np.array([[0.5, -0.2], [0.3, 0.4]], dtype=float)
    bias = np.array([0.1, -0.05], dtype=float)
    model = {
        "feature_names": ["latent_0", "latent_1", "f0", "f1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "autoencoder": {
            "input_features": ["f0", "f1"],
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "feature_names": ["latent_0", "latent_1"],
            "latent_dim": 2,
        },
        "power_transformer": {
            "features": ["latent_0", "latent_1"],
            "lambdas": [0.1, -0.2],
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
        },
    }
    return FeaturePipeline.from_model(model)


def _measure_runtime(func, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return min(timings)


def test_vectorized_transform_matches_scalar_path() -> None:
    pipeline = _make_pipeline()
    rng = np.random.default_rng(42)
    data = rng.normal(size=(128, len(pipeline.input_columns)))

    vectorized = pipeline.transform_matrix(data)
    scalar = np.vstack([pipeline.transform_array(row) for row in data])

    np.testing.assert_allclose(vectorized, scalar)


def test_vectorized_transform_is_significantly_faster() -> None:
    pipeline = _make_pipeline()
    rng = np.random.default_rng(7)
    data = rng.normal(size=(5000, len(pipeline.input_columns)))

    baseline = np.vstack([pipeline.transform_array(row) for row in data])
    vectorized = pipeline.transform_matrix(data)
    np.testing.assert_allclose(vectorized, baseline)

    # Warm up both code paths before timing.
    pipeline.transform_array(data[0])
    pipeline.transform_matrix(data[:1])

    vector_time = _measure_runtime(lambda: pipeline.transform_matrix(data))
    scalar_time = _measure_runtime(lambda: np.vstack([pipeline.transform_array(row) for row in data]))

    assert vector_time < scalar_time * 0.5
