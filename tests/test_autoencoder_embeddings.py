import json
import os
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

if "gplearn" not in sys.modules:
    gplearn_mod = types.ModuleType("gplearn")
    gplearn_mod.genetic = types.SimpleNamespace(SymbolicTransformer=object)
    sys.modules["gplearn"] = gplearn_mod
    sys.modules["gplearn.genetic"] = gplearn_mod.genetic

from botcopier.training import pipeline, preprocessing as preprocessing_mod
from botcopier.utils.inference import FeaturePipeline


@pytest.fixture()
def sample_autoencoder(tmp_path, monkeypatch):
    """Create a mocked autoencoder checkpoint and associated metadata."""

    weights = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=float)
    bias = np.array([0.1, -0.2], dtype=float)
    ae_path = tmp_path / "autoencoder.pt"
    ae_path.write_bytes(b"mock")

    def fake_extract(path: Path):
        assert Path(path) == ae_path
        return weights, bias

    monkeypatch.setattr(
        preprocessing_mod,
        "extract_torch_encoder_weights",
        fake_extract,
    )
    return ae_path, weights, bias


@pytest.fixture()
def fake_onnx_session(monkeypatch):
    """Provide a lightweight ONNXRuntime stub for nonlinear autoencoders."""

    class FakeInferenceSession:
        def __init__(self, source, providers=None):
            if isinstance(source, (bytes, bytearray)):
                raw = bytes(source) if isinstance(source, bytearray) else source
                data = json.loads(raw.decode("utf-8"))
            else:
                data = json.loads(Path(source).read_text())
            self.weights = np.asarray(data["weights"], dtype=np.float32)
            self.bias = np.asarray(data["bias"], dtype=np.float32)
            self.activation = data.get("activation", "relu")
            self.input_name = data.get("input_name", "input")
            self.output_name = data.get("output_name", "output")
            self.ops = data.get("ops", [])

        def get_inputs(self):
            return [types.SimpleNamespace(name=self.input_name)]

        def get_outputs(self):
            return [types.SimpleNamespace(name=self.output_name)]

        def run(self, output_names, feeds):
            arr = np.asarray(feeds[self.input_name], dtype=np.float32)
            result = arr @ self.weights.T + self.bias
            if self.activation == "relu":
                result = np.maximum(result, 0.0)
            return [result]

    def fake_onnx_load(path):
        data = json.loads(Path(path).read_text())

        class Graph:
            node = [types.SimpleNamespace(op_type=op) for op in data.get("ops", [])]

        return types.SimpleNamespace(graph=Graph())

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=FakeInferenceSession),
    )
    monkeypatch.setitem(sys.modules, "onnx", types.SimpleNamespace(load=fake_onnx_load))

    return FakeInferenceSession


@pytest.fixture()
def simple_session_pipeline():
    """Provide a minimal model with multiple session configurations."""

    X = np.array(
        [
            [-0.4],
            [0.0],
            [0.6],
        ],
        dtype=float,
    )
    model = {
        "feature_names": ["f0"],
        "feature_metadata": [{"original_column": "f0"}],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "clip_low": [-10.0],
        "clip_high": [10.0],
        "session_models": {
            "asian": {"coefficients": [0.8], "intercept": -0.2},
            "european": {"coefficients": [-0.5], "intercept": 0.6},
        },
    }
    return model, X


def test_autoencoder_embeddings_roundtrip(tmp_path, sample_autoencoder):
    ae_path, weights, bias = sample_autoencoder
    X = np.array(
        [
            [0.5, 0.25],
            [0.8, 0.4],
            [1.2, 0.6],
        ],
        dtype=float,
    )
    embeddings = preprocessing_mod.encode_with_autoencoder(X, ae_path, latent_dim=2)
    centered = X - X.mean(axis=0)
    expected = centered @ weights.T + bias
    np.testing.assert_allclose(embeddings, expected)
    metadata = preprocessing_mod.load_autoencoder_metadata(ae_path)
    assert metadata is not None
    metadata = dict(metadata)
    metadata.setdefault("latent_dim", int(embeddings.shape[1]))
    metadata["feature_names"] = [f"ae_{i}" for i in range(embeddings.shape[1])]
    metadata["input_features"] = ["atr", "sl_dist_atr"]
    meta_path = preprocessing_mod.save_autoencoder_metadata(ae_path, metadata)
    metadata["metadata_file"] = os.path.relpath(meta_path, ae_path.parent)
    preprocessing_mod.save_autoencoder_metadata(ae_path, metadata)
    loaded = preprocessing_mod.load_autoencoder_metadata(ae_path)
    assert loaded is not None
    np.testing.assert_allclose(
        embeddings,
        preprocessing_mod.apply_autoencoder_from_metadata(X, loaded),
    )

    coeff = np.array([0.7, -0.4], dtype=float)
    model = {
        "feature_names": loaded["feature_names"],
        "feature_metadata": [
            {"original_column": "atr"},
            {"original_column": "sl_dist_atr"},
        ],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "clip_low": [-1e6, -1e6],
        "clip_high": [1e6, 1e6],
        "coefficients": coeff.tolist(),
        "intercept": 0.0,
        "autoencoder": loaded,
    }
    preds = pipeline.predict_expected_value(model, X, model_dir=tmp_path)
    feature_pipe = FeaturePipeline.from_model(model, model_dir=tmp_path)
    transformed = feature_pipe.transform_matrix(X)
    np.testing.assert_allclose(
        transformed,
        preprocessing_mod.apply_autoencoder_from_metadata(X, loaded),
    )
    logits = transformed @ coeff
    expected_prob = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(preds, expected_prob)


def test_nonlinear_onnx_autoencoder(tmp_path, fake_onnx_session):
    onnx_path = tmp_path / "autoencoder.onnx"
    X = np.array(
        [
            [0.5, 0.25],
            [0.8, 0.4],
            [1.2, 0.6],
            [1.5, 0.75],
        ],
        dtype=float,
    )

    payload = {
        "weights": [[0.4, -0.1], [0.3, 0.2]],
        "bias": [0.05, -0.15],
        "ops": ["MatMul", "Add", "Relu"],
        "activation": "relu",
        "input_name": "input",
        "output_name": "output",
    }
    onnx_path.write_text(json.dumps(payload))

    embeddings = preprocessing_mod.encode_with_autoencoder(X, onnx_path, latent_dim=2)
    metadata = preprocessing_mod.load_autoencoder_metadata(onnx_path)
    assert metadata is not None
    assert metadata.get("format") == "onnx_nonlin"
    assert metadata.get("onnx_serialized")

    centered = X - X.mean(axis=0)
    session = fake_onnx_session(str(onnx_path))
    expected = session.run(None, {session.input_name: centered.astype(np.float32)})[0]

    np.testing.assert_allclose(embeddings, expected)
    np.testing.assert_allclose(
        preprocessing_mod.apply_autoencoder_from_metadata(X, metadata),
        expected,
    )


def test_expected_value_with_onnx_weights_file(tmp_path, fake_onnx_session):
    onnx_path = tmp_path / "autoencoder.onnx"
    X = np.array(
        [
            [0.2, -0.1],
            [1.1, 0.4],
            [-0.3, 0.9],
        ],
        dtype=float,
    )

    payload = {
        "weights": [[0.3, -0.2], [0.15, 0.25]],
        "bias": [0.05, -0.1],
        "ops": ["MatMul", "Add", "Relu"],
        "activation": "relu",
        "input_name": "features",
        "output_name": "embedding",
    }
    onnx_path.write_text(json.dumps(payload))

    embeddings = preprocessing_mod.encode_with_autoencoder(X, onnx_path, latent_dim=2)
    metadata = preprocessing_mod.load_autoencoder_metadata(onnx_path)
    assert metadata is not None
    metadata = dict(metadata)
    metadata["feature_names"] = ["latent_0", "latent_1"]
    metadata["input_features"] = ["f0", "f1"]
    preprocessing_mod.save_autoencoder_metadata(onnx_path, metadata)

    model = {
        "feature_names": ["latent_0", "latent_1"],
        "feature_metadata": [
            {"original_column": "f0"},
            {"original_column": "f1"},
        ],
        "coefficients": [0.7, -0.3],
        "intercept": 0.2,
        "clip_low": [-10.0, -10.0],
        "clip_high": [10.0, 10.0],
        "feature_mean": [0.1, -0.2],
        "feature_std": [1.5, 0.8],
        "autoencoder": {
            "weights_file": os.path.relpath(onnx_path, tmp_path),
        },
    }

    preds = pipeline.predict_expected_value(model, X, model_dir=tmp_path)

    feature_pipe = FeaturePipeline.from_model(model, model_dir=tmp_path)
    transformed = feature_pipe.transform_matrix(X)
    np.testing.assert_allclose(transformed, embeddings)

    clip_low = np.asarray(model["clip_low"], dtype=float)
    clip_high = np.asarray(model["clip_high"], dtype=float)
    clipped = np.clip(transformed, clip_low, clip_high)
    mean = np.asarray(model["feature_mean"], dtype=float)
    std = np.asarray(model["feature_std"], dtype=float)
    scaled = (clipped - mean) / np.where(std == 0, 1.0, std)
    coeff = np.asarray(model["coefficients"], dtype=float)
    logits = scaled @ coeff + float(model["intercept"])
    expected = 1.0 / (1.0 + np.exp(-logits))

    np.testing.assert_allclose(preds, expected)


def test_predict_expected_value_selects_session(simple_session_pipeline):
    model, X = simple_session_pipeline

    default_preds = pipeline.predict_expected_value(model, X)
    asian_preds = pipeline.predict_expected_value(model, X, session_key="asian")
    european_preds = pipeline.predict_expected_value(
        model, X, session_key="european"
    )

    np.testing.assert_allclose(default_preds, asian_preds)
    assert not np.allclose(asian_preds, european_preds)
    assert np.max(np.abs(asian_preds - european_preds)) > 1e-6


def test_predict_expected_value_missing_session(simple_session_pipeline):
    model, X = simple_session_pipeline

    with pytest.raises(ValueError, match="session 'unknown'"):
        pipeline.predict_expected_value(model, X, session_key="unknown")

    base_model = {k: v for k, v in model.items() if k != "session_models"}
    with pytest.raises(ValueError, match="does not define session_models"):
        pipeline.predict_expected_value(base_model, X, session_key="asian")
