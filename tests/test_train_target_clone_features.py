import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_target_clone import (
    _encode_with_autoencoder,
    _extract_features,
    _load_logs,
    _neutralize_against_market_index,
    train,
)


def test_price_indicators_persisted(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.5,0,EURUSD\n",
        "1,1.1,110,1.6,1,EURUSD\n",
        "0,1.2,120,1.7,2,EURUSD\n",
        "1,1.3,130,1.8,3,EURUSD\n",
        "0,1.4,140,1.9,4,EURUSD\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for name in [
        "sma",
        "rsi",
        "macd",
        "macd_signal",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "atr",
    ]:
        assert name in model["feature_names"]
    fft_cols = ["fft_0_mag", "fft_0_phase", "fft_1_mag", "fft_1_phase"]
    for col in fft_cols:
        assert col in model["feature_names"]
    for col in ["price", "volume", "spread"]:
        for feat in [f"{col}_lag_1", f"{col}_lag_5", f"{col}_diff"]:
            assert feat in model["feature_names"]
    df, feature_cols, _ = _load_logs(data)
    df, _, _, _ = _extract_features(df, feature_cols)
    for col in ["price", "volume", "spread"]:
        for feat in [f"{col}_lag_1", f"{col}_lag_5", f"{col}_diff"]:
            assert df[feat].notna().all()
    for col in fft_cols:
        assert np.isfinite(df[col]).all()


def test_neighbor_correlation_features(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,hour,symbol,event_time\n",
        "0,1.0,0,EURUSD,2020-01-01T00:00:00\n",
        "0,0.9,0,USDCHF,2020-01-01T00:00:00\n",
        "1,1.1,1,EURUSD,2020-01-01T00:01:00\n",
        "1,0.95,1,USDCHF,2020-01-01T00:01:00\n",
        "0,1.2,2,EURUSD,2020-01-01T00:02:00\n",
        "0,1.0,2,USDCHF,2020-01-01T00:02:00\n",
        "1,1.3,3,EURUSD,2020-01-01T00:03:00\n",
        "1,1.05,3,USDCHF,2020-01-01T00:03:00\n",
        "0,1.4,4,EURUSD,2020-01-01T00:04:00\n",
        "0,1.1,4,USDCHF,2020-01-01T00:04:00\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    sg_path = Path(__file__).resolve().parent.parent / "symbol_graph.json"
    train(data, out_dir, symbol_graph=sg_path, neighbor_corr_windows=[3])
    model = json.loads((out_dir / "model.json").read_text())
    corr_cols = ["corr_EURUSD_USDCHF_w3", "corr_USDCHF_EURUSD_w3"]
    for col in corr_cols:
        assert col in model["feature_names"]
    df, feature_cols, _ = _load_logs(data)
    df, _, _, _ = _extract_features(
        df, feature_cols, symbol_graph=sg_path, neighbor_corr_windows=[3]
    )
    for col in corr_cols:
        assert df[col].notna().all()


def test_mutual_info_feature_filter(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.0,0,EURUSD\n",
        "1,1.1,110,1.0,1,EURUSD\n",
        "0,1.2,120,1.0,2,EURUSD\n",
        "1,1.3,130,1.0,3,EURUSD\n",
    ]
    data.write_text("".join(rows))
    out_low = tmp_path / "out_low"
    train(data, out_low, mi_threshold=0.0)
    model_low = json.loads((out_low / "model.json").read_text())
    out_high = tmp_path / "out_high"
    train(data, out_high, mi_threshold=0.1)
    model_high = json.loads((out_high / "model.json").read_text())
    assert "spread_lag_1" in model_low["feature_names"]
    assert "spread_lag_1" not in model_high["feature_names"]
    assert len(model_high["feature_names"]) < len(model_low["feature_names"])


def test_autoencoder_embedding_shapes(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.5,0,EURUSD\n",
        "1,1.1,110,1.6,1,EURUSD\n",
        "0,1.2,120,1.7,2,EURUSD\n",
        "1,1.3,130,1.8,3,EURUSD\n",
        "0,1.4,140,1.9,4,EURUSD\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    train(
        data,
        out_dir,
        use_autoencoder=True,
        autoencoder_dim=2,
        autoencoder_epochs=5,
    )
    assert (out_dir / "autoencoder.pt").exists()
    model = json.loads((out_dir / "model.json").read_text())
    assert model["feature_names"] == ["ae_0", "ae_1"]
    df, feature_cols, _ = _load_logs(data)
    df, feature_cols, _, _ = _extract_features(df, feature_cols)
    X = df[feature_cols].to_numpy(dtype=float)
    emb = _encode_with_autoencoder(X, out_dir / "autoencoder.pt")
    assert emb.shape == (len(df), 2)


def test_news_sentiment_feature_join(tmp_path: Path) -> None:
    trades = tmp_path / "trades_raw.csv"
    trades_rows = [
        "label,price,hour,symbol,event_time\n",
        "0,1.0,0,EURUSD,2020-01-01T00:00:00\n",
        "1,1.1,1,EURUSD,2020-01-01T01:00:00\n",
    ]
    trades.write_text("".join(trades_rows))

    sentiment = tmp_path / "news_sentiment.csv"
    sentiment_rows = [
        "symbol,timestamp,score\n",
        "EURUSD,2020-01-01T00:30:00,0.5\n",
    ]
    sentiment.write_text("".join(sentiment_rows))

    df, feature_cols, _ = _load_logs(trades)
    ns_df = pd.read_csv(sentiment)
    df, feature_cols, _, _ = _extract_features(df, feature_cols, news_sentiment=ns_df)
    assert "sentiment_score" in feature_cols
    assert df["sentiment_score"].notna().all()


def test_market_index_neutralization(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,hour,symbol\n",
        "0,1.0,0,EURUSD\n",
        "0,0.9,0,USDCHF\n",
        "1,1.1,1,EURUSD\n",
        "1,0.95,1,USDCHF\n",
        "0,1.2,2,EURUSD\n",
        "0,1.0,2,USDCHF\n",
    ]
    data.write_text("".join(rows))
    df, feature_cols, _ = _load_logs(data)
    df, feature_cols, _, _ = _extract_features(df, feature_cols)
    df_before = df.copy()
    df_neu, feature_cols_neu = _neutralize_against_market_index(df, feature_cols)

    price = df_before["price"]
    returns = price.groupby(df_before["symbol"]).pct_change().fillna(0.0)
    market_idx = returns.groupby(df_before.index).transform("mean")
    feat = "price_lag_1"
    assert feat in feature_cols_neu
    corr_before = float(np.corrcoef(df_before[feat], market_idx)[0, 1])
    corr_after = float(np.corrcoef(df_neu[feat], market_idx)[0, 1])
    var_before = float(df_before[feat].var())
    var_after = float(df_neu[feat].var())
    assert abs(corr_after) < abs(corr_before) or np.isclose(corr_before, 0)
    assert var_after <= var_before

