import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botcopier.models.deep import CrossModalTransformer, TemporalConvNet
from botcopier.models.registry import (
    TabTransformer,
    fit_crossmodal_transformer,
    fit_multi_symbol_attention,
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


def test_multi_symbol_attention_improves_accuracy():
    rng = np.random.default_rng(42)
    n_samples = 400
    symbols = np.array(["EURUSD", "USDCHF"] * (n_samples // 2))
    signal = rng.normal(size=n_samples)
    y = np.where(symbols == "EURUSD", signal > 0, signal < 0).astype(float)
    X = signal.reshape(-1, 1)

    graph_path = Path("symbol_graph.json")
    graph = json.loads(graph_path.read_text())
    symbol_names = graph.get("symbols", [])
    embeddings_map = graph.get("embeddings", {})
    embeddings = np.array([embeddings_map[name] for name in symbol_names], dtype=float)
    edge_index = graph.get("edge_index", [[], []])
    neighbor_lists: list[list[int]] = [[] for _ in symbol_names]
    for src, dst in zip(edge_index[0], edge_index[1]):
        neighbor_lists[int(src)].append(int(dst))
    for idx in range(len(symbol_names)):
        if idx not in neighbor_lists[idx]:
            neighbor_lists[idx].insert(0, idx)

    symbol_to_idx = {name: i for i, name in enumerate(symbol_names)}
    symbol_ids = np.array([symbol_to_idx[sym] for sym in symbols], dtype=int)

    meta, predict = fit_multi_symbol_attention(
        X,
        y,
        symbol_ids=symbol_ids,
        symbol_names=symbol_names,
        embeddings=embeddings,
        neighbor_index=neighbor_lists,
        epochs=30,
        batch_size=64,
        lr=5e-3,
        dropout=0.0,
        hidden_dim=32,
        heads=2,
        patience=5,
        mixed_precision=False,
    )
    preds = predict((X, symbol_ids))
    acc = ((preds >= 0.5).astype(float) == y).mean()
    from sklearn.linear_model import LogisticRegression

    baseline = LogisticRegression().fit(X, y).score(X, y)
    assert acc > baseline + 0.1
    assert meta["attention_weights"]["EURUSD"]
    assert meta["neighbor_order"]["EURUSD"][0] == "EURUSD"


__all__ = [
    "test_tabtransformer_forward_and_accuracy",
    "test_temporal_convnet_forward_and_accuracy",
    "test_crossmodal_forward_and_training",
    "test_multi_symbol_attention_improves_accuracy",
]
