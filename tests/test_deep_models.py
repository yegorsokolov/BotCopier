import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botcopier.models.deep import CrossModalTransformer, TemporalConvNet
from botcopier.models.registry import (
    TabTransformer,
    fit_crossmodal_transformer,
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


def test_crossmodal_forward_and_training():
    rng = np.random.default_rng(2)
    price = rng.normal(size=(64, 3, 4))
    news = rng.normal(size=(64, 2, 5))
    y = (price[:, :, 0].mean(axis=1) + 0.5 * news[:, :, 0].mean(axis=1) > 0).astype(float)

    torch.manual_seed(0)
    net = CrossModalTransformer(price_features=4, news_features=5, price_window=3, news_window=2)
    price_t = torch.tensor(price, dtype=torch.float32)
    news_t = torch.tensor(news, dtype=torch.float32)
    logits = net(price_t, news_t)
    assert logits.shape == (price.shape[0],)

    meta, predict = fit_crossmodal_transformer(
        (price, news),
        y,
        epochs=6,
        batch_size=16,
        lr=5e-3,
        dropout=0.0,
        mixed_precision=False,
    )
    preds = predict((price, news))
    acc = ((preds >= 0.5).astype(float) == y).mean()
    assert acc > 0.6

    arch = meta.get("architecture", {})
    assert arch.get("type") == "CrossModalTransformer"
    assert arch.get("price_features") == 4
    assert arch.get("news_features") == 5
    assert arch.get("price_window") == 3
    assert arch.get("news_window") == 2
    assert "news_clip_low" in meta


__all__ = [
    "test_tabtransformer_forward_and_accuracy",
    "test_temporal_convnet_forward_and_accuracy",
    "test_crossmodal_forward_and_training",
]
