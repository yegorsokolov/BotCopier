"""Unit tests covering autoencoder output mapping in the feature pipeline."""

import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


if "gplearn" not in sys.modules:
    gplearn_mod = types.ModuleType("gplearn")
    gplearn_mod.genetic = types.SimpleNamespace(SymbolicTransformer=object)
    sys.modules["gplearn"] = gplearn_mod
    sys.modules["gplearn.genetic"] = gplearn_mod.genetic


def _ensure_preprocessing_stub() -> None:
    if "botcopier.training.preprocessing" in sys.modules:
        return

    try:  # Prefer the real implementation when available.
        import botcopier.training.preprocessing  # type: ignore  # noqa: F401
        return
    except Exception:  # pragma: no cover - fallback to lightweight stub
        pass

    preprocessing_stub = types.ModuleType("botcopier.training.preprocessing")

    def apply_autoencoder_from_metadata(X: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mean = np.asarray(metadata.get("input_mean", []), dtype=float)
        scale = np.asarray(metadata.get("input_scale", []), dtype=float)
        if mean.size and mean.shape[0] == data.shape[1]:
            data = data - mean
        if scale.size and scale.shape[0] == data.shape[1]:
            safe_scale = np.where(scale == 0, 1.0, scale)
            data = data / safe_scale
        weights = np.asarray(metadata.get("weights", []), dtype=float)
        bias = np.asarray(metadata.get("bias", []), dtype=float)
        embedding = data @ weights.T if weights.size else np.zeros((data.shape[0], 0))
        if bias.size:
            embedding = embedding + bias
        return embedding.astype(float)

    def load_autoencoder_metadata(path):  # pragma: no cover - compatibility stub
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return None

    preprocessing_stub.apply_autoencoder_from_metadata = apply_autoencoder_from_metadata
    preprocessing_stub.load_autoencoder_metadata = load_autoencoder_metadata

    training_stub = types.ModuleType("botcopier.training")
    training_stub.__path__ = []  # type: ignore[attr-defined]
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


def test_feature_pipeline_autoencoder_schema_handles_alias_metadata() -> None:
    weights = np.array([[0.4, -0.2], [0.3, 0.6]], dtype=float)
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
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "feature_names": ["latent_0", "latent_1"],
        },
    }

    pipeline_obj = FeaturePipeline.from_model(model)

    assert pipeline_obj.autoencoder_inputs == ["f0", "f1"]
    assert pipeline_obj.input_columns == ["f0", "f1"]
    assert pipeline_obj.schema_columns == ["f0", "f1"]

    matrix = np.array(
        [
            [0.2, -0.1],
            [1.0, 0.4],
            [-0.6, 0.9],
        ],
        dtype=float,
    )

    transformed = pipeline_obj.transform_matrix(matrix)
    expected_latent = matrix @ weights.T + bias
    np.testing.assert_allclose(transformed[:, :2], expected_latent)
    np.testing.assert_allclose(transformed[:, 2:], matrix)


def test_feature_pipeline_autoencoder_subset_inputs_validation() -> None:
    pd = pytest.importorskip("pandas")
    from botcopier.data.feature_schema import FeatureSchema
    from botcopier.training.pipeline import predict_expected_value

    weights = np.array([[0.5, -0.15], [0.25, 0.35]], dtype=float)
    bias = np.array([0.05, -0.1], dtype=float)
    coefficients = np.array([0.4, -0.3, 0.2], dtype=float)
    intercept = 0.15

    model = {
        "feature_names": ["latent_0", "latent_1", "book_imbalance"],
        "feature_metadata": [
            {"original_column": "atr"},
            {"original_column": "sl_dist_atr"},
            {"original_column": "book_imbalance"},
        ],
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "autoencoder": {
            "input_features": ["atr", "sl_dist_atr"],
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "feature_names": ["latent_0", "latent_1"],
        },
    }

    pipeline_obj = FeaturePipeline.from_model(model)

    assert pipeline_obj.autoencoder_inputs == ["atr", "sl_dist_atr"]
    assert pipeline_obj.input_columns == ["atr", "sl_dist_atr", "book_imbalance"]
    assert pipeline_obj.schema_columns == pipeline_obj.input_columns

    matrix = np.array(
        [
            [0.8, 1.5, 0.25],
            [0.5, 0.9, -0.4],
            [0.3, 0.7, 0.1],
        ],
        dtype=float,
    )

    df = pd.DataFrame(matrix, columns=pipeline_obj.schema_columns)
    # Ensure Pandera validation succeeds when extra non-autoencoder columns are
    # present alongside the autoencoder inputs.
    FeatureSchema.validate(df, lazy=True)

    transformed = pipeline_obj.transform_matrix(matrix)
    latent = matrix[:, :2] @ weights.T + bias
    expected = np.concatenate([latent, matrix[:, 2:].reshape(matrix.shape[0], 1)], axis=1)
    np.testing.assert_allclose(transformed, expected)

    preds = predict_expected_value(model, matrix)
    logits = expected @ coefficients + intercept
    expected_prob = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(preds, expected_prob)
