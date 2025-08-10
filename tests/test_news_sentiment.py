import csv
import json
from datetime import datetime
from pathlib import Path

import scripts.fetch_news as fetch_news
from scripts.train_target_clone import train


def _write_sample_log(file: Path):
    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
        "open_time",
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "sl_hit_dist",
        "tp_hit_dist",
    ]
    rows = [
        [
            "1",
            "2024.01.01 00:00:00",
            "",
            "",
            "OPEN",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "2",
            "",
            "0.1",
            "0.0001",
            "100",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
        ],
        [
            "2",
            "2024.01.01 01:00:00",
            "",
            "",
            "OPEN",
            "2",
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.2000",
            "1.1950",
            "1.2100",
            "0",
            "3",
            "",
            "0.1",
            "0.0002",
            "200",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_news_sentiment_in_model(tmp_path, monkeypatch):
    class DummyPipe:
        def __call__(self, text):
            return [{"label": "positive", "score": 0.8}]

    monkeypatch.setattr(fetch_news, "_get_pipeline", lambda: DummyPipe())
    score = fetch_news.compute_sentiment(["Great earnings for EURUSD"])
    assert score > 0

    sent_file = tmp_path / "news.csv"
    with open(sent_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "score"])
        writer.writerow([datetime(2024, 1, 1).isoformat(), "EURUSD", score])

    data_dir = tmp_path / "logs"
    data_dir.mkdir()
    _write_sample_log(data_dir / "trades_sample.csv")

    out_dir = tmp_path / "out"
    train(data_dir, out_dir, news_sentiment_file=sent_file)

    model_file = out_dir / "model.json"
    with open(model_file) as f:
        model = json.load(f)
    assert "news_sentiment" in model.get("feature_names", [])
