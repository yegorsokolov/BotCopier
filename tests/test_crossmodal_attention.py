import json
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.train_target_clone import train


def test_crossmodal_attention_shape(tmp_path: Path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,event_time,symbol\n"
        "0,1.0,2020-01-01T00:00:00Z,AAA\n"
        "1,1.1,2020-01-01T00:01:00Z,AAA\n"
        "0,1.2,2020-01-01T00:02:00Z,AAA\n"
        "1,1.3,2020-01-01T00:03:00Z,AAA\n"
        "0,1.4,2020-01-01T00:04:00Z,AAA\n"
        "1,1.5,2020-01-01T00:05:00Z,AAA\n"
    )
    ns = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="T"),
            "symbol": ["AAA"] * 6,
            "score": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    out_dir = tmp_path / "out"
    model = train(
        data,
        out_dir,
        model_type="crossmodal",
        window=2,
        epochs=1,
        news_sentiment=ns,
    )
    price_dim = model.price_proj.in_features
    news_dim = model.news_proj.in_features
    dummy_price = torch.randn(1, 2, price_dim)
    dummy_news = torch.randn(1, 2, news_dim)
    logits, attn = model(dummy_price, dummy_news)
    assert logits.shape == (1, 1)
    assert attn.shape == (1, 2, 2)
    meta = json.loads((out_dir / "model.json").read_text())
    assert meta["model_type"] == "crossmodal"
    assert meta["feature_names"]
    assert meta["sentiment_feature"] == "sentiment_score"
