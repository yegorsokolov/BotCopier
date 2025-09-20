import json
import os
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.SimpleNamespace()

from botcopier.training import pipeline


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
        pipeline,
        "_extract_torch_encoder_weights",
        fake_extract,
    )
    return ae_path, weights, bias


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
    embeddings = pipeline._encode_with_autoencoder(X, ae_path, latent_dim=2)
    centered = X - X.mean(axis=0)
    expected = centered @ weights.T + bias
    np.testing.assert_allclose(embeddings, expected)
    metadata = pipeline._load_autoencoder_metadata(ae_path)
    assert metadata is not None
    metadata = dict(metadata)
    metadata.setdefault("latent_dim", int(embeddings.shape[1]))
    metadata["feature_names"] = [f"ae_{i}" for i in range(embeddings.shape[1])]
    metadata["input_features"] = ["atr", "sl_dist_atr"]
    meta_path = pipeline._save_autoencoder_metadata(ae_path, metadata)
    metadata["metadata_file"] = os.path.relpath(meta_path, ae_path.parent)
    pipeline._save_autoencoder_metadata(ae_path, metadata)
    loaded = pipeline._load_autoencoder_metadata(ae_path)
    assert loaded is not None
    np.testing.assert_allclose(
        embeddings,
        pipeline._apply_autoencoder_from_metadata(X, loaded),
    )

    coeff = np.array([0.7, -0.4], dtype=float)
    model = {
        "feature_names": loaded["feature_names"],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "clip_low": [-1e6, -1e6],
        "clip_high": [1e6, 1e6],
        "coefficients": coeff.tolist(),
        "intercept": 0.0,
        "autoencoder": loaded,
    }
    preds = pipeline.predict_expected_value(model, X)
    expected_features = pipeline._apply_autoencoder_from_metadata(X, loaded)
    logits = expected_features @ coeff
    expected_prob = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(preds, expected_prob)


def test_nonlinear_onnx_autoencoder(tmp_path, monkeypatch):
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

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(InferenceSession=FakeInferenceSession))
    monkeypatch.setitem(sys.modules, "onnx", types.SimpleNamespace(load=fake_onnx_load))

    embeddings = pipeline._encode_with_autoencoder(X, onnx_path, latent_dim=2)
    metadata = pipeline._load_autoencoder_metadata(onnx_path)
    assert metadata is not None
    assert metadata.get("format") == "onnx_nonlin"
    assert metadata.get("onnx_serialized")

    centered = X - X.mean(axis=0)
    session = FakeInferenceSession(str(onnx_path))
    expected = session.run(None, {session.input_name: centered.astype(np.float32)})[0]

    np.testing.assert_allclose(embeddings, expected)
    np.testing.assert_allclose(
        pipeline._apply_autoencoder_from_metadata(X, metadata),
        expected,
    )
