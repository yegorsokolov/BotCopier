import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

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
