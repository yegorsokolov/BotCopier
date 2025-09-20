import base64
import importlib
import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from botcopier.models.registry import get_model
from botcopier.utils.inference import FeaturePipeline


def _build_gradient_boosting_checkpoint() -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 3))
    logits = 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2]
    y = (logits > 0).astype(int)
    builder = get_model("gradient_boosting")
    meta, _ = builder(X, y, random_state=0)
    payload = meta["gb_model"]
    model = {
        "feature_names": [f"f{i}" for i in range(X.shape[1])],
        "feature_metadata": [{"original_column": f"f{i}"} for i in range(X.shape[1])],
        "gb_model": payload,
        "threshold": 0.0,
    }
    estimator = pickle.loads(base64.b64decode(payload))
    expected = estimator.predict_proba(X)[:, 1]
    return model, X, expected


def _feature_dict(row: np.ndarray) -> dict[str, float]:
    return {f"f{i}": float(value) for i, value in enumerate(row)}


def test_serve_model_handles_gradient_boosting(tmp_path: Path) -> None:
    model, X, expected = _build_gradient_boosting_checkpoint()
    serve_module = importlib.reload(importlib.import_module("botcopier.scripts.serve_model"))
    serve_module.MODEL_DIR = tmp_path
    serve_module._configure_model(model)
    preds = np.array([serve_module._predict_one(row.tolist()) for row in X])
    np.testing.assert_allclose(preds, expected, rtol=1e-6, atol=1e-6)


def test_replay_script_handles_gradient_boosting(tmp_path: Path) -> None:
    model, X, expected = _build_gradient_boosting_checkpoint()
    replay_module = importlib.reload(importlib.import_module("botcopier.scripts.replay_decisions"))
    replay_module.MODEL_DIR = tmp_path
    replay_module.FEATURE_PIPELINE = FeaturePipeline.from_model(model, model_dir=tmp_path)
    probs = [replay_module._predict_gradient_boosting(model, _feature_dict(row)) for row in X]
    np.testing.assert_allclose(probs, expected, rtol=1e-6, atol=1e-6)


def test_grpc_predict_server_handles_gradient_boosting(tmp_path: Path) -> None:
    model, X, expected = _build_gradient_boosting_checkpoint()
    grpc_module = importlib.reload(importlib.import_module("botcopier.scripts.grpc_predict_server"))
    grpc_module.MODEL_DIR = tmp_path
    grpc_module._configure_runtime(model)
    preds = np.array([grpc_module._predict_one(row.tolist()) for row in X])
    np.testing.assert_allclose(preds, expected, rtol=1e-6, atol=1e-6)
