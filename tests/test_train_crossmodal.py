import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from botcopier.training.pipeline import train


def test_crossmodal_training_with_news(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "event_time,symbol,price,profit\n",
        "2024-01-01T00:00:00Z,EURUSD,1.1000,1.0\n",
        "2024-01-01T00:01:00Z,EURUSD,1.1010,-0.5\n",
        "2024-01-01T00:02:00Z,EURUSD,1.1020,1.2\n",
        "2024-01-01T00:03:00Z,EURUSD,1.1030,-0.7\n",
    ]
    data.write_text("".join(rows))
    news = tmp_path / "news_embeddings.csv"
    news_rows = [
        "symbol,timestamp,emb0,emb1\n",
        "EURUSD,2024-01-01T00:00:30Z,0.1,0.2\n",
        "EURUSD,2024-01-01T00:02:15Z,-0.3,0.4\n",
    ]
    news.write_text("".join(news_rows))
    out_dir = tmp_path / "out"
    model_obj = train(
        data,
        out_dir,
        model_type="crossmodal",
        window=2,
        news_window=2,
        news_horizon_seconds=300.0,
        epochs=1,
        batch_size=2,
    )
    assert next(model_obj.parameters()).device.type == "cpu"
    model_json = json.loads((out_dir / "model.json").read_text())
    assert model_json["model_type"] == "crossmodal"
    assert model_json["architecture"]["type"] == "CrossModalTransformer"
    news_meta = model_json.get("news_embeddings")
    assert news_meta
    assert news_meta["window"] == 2
    assert news_meta["dimension"] == 2
    assert "news_clip_low" in model_json
    assert "news_clip_high" in model_json
