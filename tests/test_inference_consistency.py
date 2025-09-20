import importlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from botcopier.training import pipeline
from botcopier.utils.inference import FeaturePipeline


def test_inference_matches_pipeline_for_autoencoder_and_power_transform(tmp_path: Path) -> None:
    weights = np.array([[0.5, -0.2], [0.3, 0.4]], dtype=float)
    bias = np.array([0.1, -0.05], dtype=float)
    model = {
        "feature_names": ["latent_0", "latent_1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "coefficients": [0.4, -0.25],
        "intercept": -0.1,
        "clip_low": [-10.0, -10.0],
        "clip_high": [10.0, 10.0],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "autoencoder": {
            "input_features": ["f0", "f1"],
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "feature_names": ["latent_0", "latent_1"],
            "latent_dim": 2,
        },
        "power_transformer": {
            "features": ["latent_1"],
            "lambdas": [0.2],
            "mean": [0.1],
            "scale": [1.5],
        },
        "threshold": 0.0,
    }

    X = np.array([[0.5, 0.3], [1.2, -0.4], [-0.1, 0.8]], dtype=float)

    expected = pipeline.predict_expected_value(model, X)

    serve_module = importlib.reload(importlib.import_module("botcopier.scripts.serve_model"))
    serve_module.MODEL_DIR = tmp_path
    serve_module._configure_model(model)
    serve_preds = np.array([serve_module._predict_one(row.tolist()) for row in X])
    np.testing.assert_allclose(serve_preds, expected)

    replay_module = importlib.reload(importlib.import_module("botcopier.scripts.replay_decisions"))
    replay_module.MODEL_DIR = tmp_path
    replay_module.FEATURE_PIPELINE = FeaturePipeline.from_model(model, model_dir=tmp_path)
    replay_preds = np.array(
        [
            replay_module._predict_logistic(
                model, {"f0": float(row[0]), "f1": float(row[1])}
            )
            for row in X
        ]
    )
    np.testing.assert_allclose(replay_preds, expected)
