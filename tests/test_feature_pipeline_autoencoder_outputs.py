"""Unit tests covering autoencoder output mapping in the feature pipeline."""

import sys
import types

import numpy as np
import pytest


def _ensure_preprocessing_stub() -> None:
    if "botcopier.training.preprocessing" in sys.modules:
        return

    preprocessing_stub = types.ModuleType("botcopier.training.preprocessing")

    def apply_autoencoder_from_metadata(X: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        weights = np.asarray(metadata.get("weights", []), dtype=float)
        bias = np.asarray(metadata.get("bias", []), dtype=float)
        embedding = data @ weights.T if weights.size else np.zeros((data.shape[0], 0))
        if bias.size:
            embedding = embedding + bias
        return embedding.astype(float)

    def load_autoencoder_metadata(path):  # pragma: no cover - not needed in tests
        return None

    preprocessing_stub.apply_autoencoder_from_metadata = apply_autoencoder_from_metadata
    preprocessing_stub.load_autoencoder_metadata = load_autoencoder_metadata

    training_stub = types.ModuleType("botcopier.training")
    training_stub.preprocessing = preprocessing_stub
    sys.modules["botcopier.training"] = training_stub
    sys.modules["botcopier.training.preprocessing"] = preprocessing_stub


_ensure_preprocessing_stub()

from botcopier.utils.inference import FeaturePipeline


def test_feature_pipeline_requires_autoencoder_outputs() -> None:
    weights = np.array([[0.6, -0.2], [0.1, 0.3]], dtype=float)
    model = {
        "feature_names": ["latent_0", "latent_1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "autoencoder": {
            "input_features": ["f0", "f1"],
            "weights": weights.tolist(),
            "bias": [0.0, 0.0],
            "latent_dim": 2,
        },
    }

    with pytest.raises(ValueError, match="output feature names"):
        FeaturePipeline.from_model(model)


def test_feature_pipeline_places_latents_in_expected_positions() -> None:
    weights = np.array([[0.4, -0.3], [0.2, 0.5]], dtype=float)
    bias = np.array([0.05, -0.15], dtype=float)
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
    }

    pipeline_obj = FeaturePipeline.from_model(model)

    matrix = np.array(
        [
            [0.5, -0.2],
            [1.2, 0.3],
            [-0.4, 0.7],
        ],
        dtype=float,
    )

    transformed = pipeline_obj.transform_matrix(matrix)
    expected_latent = matrix @ weights.T + bias

    np.testing.assert_allclose(transformed[:, :2], expected_latent)
    np.testing.assert_allclose(transformed[:, 2:], matrix)

    mapping_result = pipeline_obj.transform_dict({"f0": 0.5, "f1": -0.2})
    np.testing.assert_allclose(
        mapping_result,
        np.concatenate([expected_latent[0], matrix[0]]),
    )
