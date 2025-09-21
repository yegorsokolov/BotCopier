import importlib
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

from botcopier.training import pipeline
from botcopier.utils.inference import FeaturePipeline


def _compute_pnl_predictions(model: dict, X: np.ndarray, model_dir: Path) -> np.ndarray:
    pipeline_obj = FeaturePipeline.from_model(model, model_dir=model_dir)
    features = pipeline_obj.transform_matrix(X)
    clip_low = np.asarray(model.get("clip_low", []), dtype=float)
    clip_high = np.asarray(model.get("clip_high", []), dtype=float)
    if clip_low.size and clip_high.size and clip_low.shape == clip_high.shape:
        features = np.clip(features, clip_low, clip_high)
    mean = np.asarray(model.get("feature_mean", []), dtype=float)
    std = np.asarray(model.get("feature_std", []), dtype=float)
    if mean.size and std.size and mean.shape == std.shape:
        denom = np.where(std == 0, 1.0, std)
        features = (features - mean) / denom
    pnl_model = model.get("pnl_model")
    if not pnl_model:
        return np.ones(features.shape[0], dtype=float)
    pnl_coef = np.asarray(pnl_model.get("coefficients", []), dtype=float)
    pnl_intercept = float(pnl_model.get("intercept", 0.0))
    return features @ pnl_coef + pnl_intercept


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

    expected = pipeline.predict_expected_value(model, X, model_dir=tmp_path)

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


def test_predict_expected_value_applies_logistic_calibration(tmp_path: Path) -> None:
    model = {
        "feature_names": ["f0", "f1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "coefficients": [0.9, -0.35],
        "intercept": -0.15,
        "clip_low": [-5.0, -5.0],
        "clip_high": [5.0, 5.0],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "calibration_coef": 1.4,
        "calibration_intercept": -0.25,
        "pnl_model": {
            "coefficients": [0.15, 0.3],
            "intercept": 0.8,
        },
    }
    X = np.array(
        [
            [0.6, -0.4],
            [1.2, 0.35],
            [-0.25, 0.7],
            [0.8, -1.1],
        ],
        dtype=float,
    )

    expected = pipeline.predict_expected_value(model, X, model_dir=tmp_path)
    baseline = dict(model)
    baseline.pop("calibration_coef", None)
    baseline.pop("calibration_intercept", None)
    uncalibrated = pipeline.predict_expected_value(baseline, X, model_dir=tmp_path)
    assert not np.allclose(expected, uncalibrated)

    replay_module = importlib.reload(
        importlib.import_module("botcopier.scripts.replay_decisions")
    )
    replay_module.MODEL_DIR = tmp_path
    replay_module.FEATURE_PIPELINE = FeaturePipeline.from_model(model, model_dir=tmp_path)
    replay_probs = np.array(
        [
            replay_module._predict_logistic(
                model, {"f0": float(row[0]), "f1": float(row[1])}
            )
            for row in X
        ],
        dtype=float,
    )

    pnl = _compute_pnl_predictions(model, X, tmp_path)
    np.testing.assert_allclose(expected, replay_probs * pnl)


def test_predict_expected_value_applies_isotonic_calibration(tmp_path: Path) -> None:
    model = {
        "feature_names": ["f0", "f1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "coefficients": [0.75, 0.5],
        "intercept": -0.05,
        "clip_low": [-5.0, -5.0],
        "clip_high": [5.0, 5.0],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "calibration": {
            "method": "isotonic",
            "x": [0.0, 0.35, 0.6, 1.0],
            "y": [0.0, 0.25, 0.85, 1.0],
        },
        "pnl_model": {
            "coefficients": [0.05, -0.12],
            "intercept": 0.9,
        },
    }
    X = np.array(
        [
            [-0.1, 0.4],
            [0.9, -0.3],
            [0.5, 0.2],
            [-0.6, 0.1],
        ],
        dtype=float,
    )

    expected = pipeline.predict_expected_value(model, X, model_dir=tmp_path)
    baseline = dict(model)
    baseline.pop("calibration", None)
    uncalibrated = pipeline.predict_expected_value(baseline, X, model_dir=tmp_path)
    assert not np.allclose(expected, uncalibrated)

    replay_module = importlib.reload(
        importlib.import_module("botcopier.scripts.replay_decisions")
    )
    replay_module.MODEL_DIR = tmp_path
    replay_module.FEATURE_PIPELINE = FeaturePipeline.from_model(model, model_dir=tmp_path)
    replay_probs = np.array(
        [
            replay_module._predict_logistic(
                model, {"f0": float(row[0]), "f1": float(row[1])}
            )
            for row in X
        ],
        dtype=float,
    )

    pnl = _compute_pnl_predictions(model, X, tmp_path)
    np.testing.assert_allclose(expected, replay_probs * pnl)
