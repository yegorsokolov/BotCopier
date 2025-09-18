from pathlib import Path

import pandas as pd

from botcopier.scripts.backtest_sentiment_embeddings import run_backtests


def test_sentiment_embedding_backtest(tmp_path: Path) -> None:
    trades = tmp_path / "trades_raw.csv"
    trades.write_text(
        "label,price,event_time,symbol\n"
        "1,1.0,2020-01-01T00:00:00Z,AAA\n"
        "0,1.0,2020-01-01T00:00:30Z,BBB\n"
        "1,1.0,2020-01-01T00:01:00Z,AAA\n"
        "0,1.0,2020-01-01T00:01:30Z,BBB\n"
        "1,1.0,2020-01-01T00:02:00Z,AAA\n"
        "0,1.0,2020-01-01T00:02:30Z,BBB\n"
        "1,1.0,2020-01-01T00:03:00Z,AAA\n"
        "0,1.0,2020-01-01T00:03:30Z,BBB\n"
    )

    news = tmp_path / "news_sentiment.csv"
    news_df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "sentiment_timestamp": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "sentiment_headline_count": [1, 1],
            "sentiment_emb_0": [1.0, -1.0],
            "sentiment_emb_1": [-1.0, 1.0],
        }
    )
    news_df.to_csv(news, index=False)

    out_dir = tmp_path / "backtests"
    result = run_backtests(trades, news, out_dir, random_seed=0)
    assert result["embedding"]["cv_accuracy"] >= result["scalar"]["cv_accuracy"]
    assert result["delta_accuracy"] >= 0
    assert (out_dir / "sentiment_backtest.json").exists()
