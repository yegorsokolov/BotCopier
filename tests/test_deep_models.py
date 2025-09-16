import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botcopier.models.deep import TemporalConvNet
from botcopier.models.registry import (
    TabTransformer,
    fit_tab_transformer,
    fit_temporal_cnn,
)


def test_tabtransformer_forward_and_accuracy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)

    torch.manual_seed(0)
    model = TabTransformer(num_features=4)
    batch = torch.tensor(X, dtype=torch.float32)
    logits = model(batch)
    assert logits.shape == (X.shape[0],)

    meta, predict = fit_tab_transformer(
        X,
        y,
        epochs=12,
        batch_size=32,
        lr=5e-3,
        dropout=0.0,
        mixed_precision=False,
    )
    preds = predict(X)
    acc = ((preds >= 0.5).astype(float) == y).mean()
    assert acc > 0.75

    arch = meta.get("architecture", {})
    assert arch.get("type") == "TabTransformer"
    assert arch.get("num_features") == 4
    assert arch.get("window") == 1
    assert "state_dict" in meta and meta["state_dict"]


def test_temporal_convnet_forward_and_accuracy():
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(96, 5, 3))
    y = (seq[:, :, 0].mean(axis=1) > 0).astype(float)

    torch.manual_seed(0)
    net = TemporalConvNet(3, [4, 4])
    inp = torch.tensor(seq.transpose(0, 2, 1), dtype=torch.float32)
    out = net(inp)
    assert out.shape == (seq.shape[0], 4, seq.shape[1])

    meta, predict = fit_temporal_cnn(
        seq,
        y,
        epochs=15,
        batch_size=24,
        lr=5e-3,
        dropout=0.0,
        channels=(8, 8),
        mixed_precision=False,
    )
    preds = predict(seq)
    acc = ((preds >= 0.5).astype(float) == y).mean()
    assert acc > 0.7

    arch = meta.get("architecture", {})
    assert arch.get("type") == "TemporalConvNet"
    assert arch.get("num_features") == 3
    assert arch.get("window") == 5
    assert "state_dict" in meta and meta["state_dict"]


__all__ = [
    "test_tabtransformer_forward_and_accuracy",
    "test_temporal_convnet_forward_and_accuracy",
]
