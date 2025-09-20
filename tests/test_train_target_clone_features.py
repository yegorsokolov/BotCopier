import json
from pathlib import Path

import numpy as np
import pandas as pd

from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.training.pipeline import (
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
    feature_config = configure_cache(FeatureConfig())
    df, feature_cols, _ = _load_logs(data, feature_config=feature_config)
    df, _, _, _ = _extract_features(df, feature_cols, config=feature_config)
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
    factor_cols = ["factor_0", "factor_1"]
    for col in factor_cols:
        assert col in model["feature_names"]
    assert "pca_components" in model
    feature_config = configure_cache(FeatureConfig())
    df, feature_cols, _ = _load_logs(data, feature_config=feature_config)
    df, _, _, _ = _extract_features(
        df,
        feature_cols,
        symbol_graph=sg_path,
        neighbor_corr_windows=[3],
        config=feature_config,
    )
    for col in corr_cols:
        assert df[col].notna().all()
        assert df[col].between(-1, 1).all()
        assert not np.allclose(df[col].to_numpy(), 0.0)
    for col in factor_cols:
        assert np.isfinite(df[col]).all()


def test_calendar_fields_utc_and_ranges() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 0],
            "price": [1.0, 1.1],
            "event_time": [
                "2024-01-01T01:00:00+02:00",
                "2024-06-01T13:30:00-05:00",
            ],
        }
    )
    feature_cols = ["price"]
    config = configure_cache(FeatureConfig())
    df, feature_cols, _, _ = _extract_features(df, feature_cols, config=config)
    assert df["hour"].tolist() == [23, 18]
    assert df["dayofweek"].tolist() == [6, 5]
    assert df["month"].tolist() == [12, 6]
    assert df["hour"].between(0, 23).all()
    assert df["dayofweek"].between(0, 6).all()
    assert df["month"].between(1, 12).all()
    for col in [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]:
        assert col in feature_cols
        assert ((df[col] >= -1) & (df[col] <= 1)).all()


def test_rank_feature_bounds() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0],
            "price": [1.0, 1.0, 2.0, 1.9],
            "volume": [100, 110, 200, 150],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "event_time": [
                "2020-01-01T00:00:00",
                "2020-01-01T00:00:00",
                "2020-01-01T00:01:00",
                "2020-01-01T00:01:00",
            ],
        }
    )
    feature_cols = ["price", "volume"]
    config = configure_cache(FeatureConfig())
    df, feature_cols, _, _ = _extract_features(
        df, feature_cols, rank_features=True, config=config
    )
    assert "ret_rank" in feature_cols
    assert "vol_rank" in feature_cols
    assert df["ret_rank"].between(0, 1).all()
    assert df["vol_rank"].between(0, 1).all()


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
    feature_config = configure_cache(FeatureConfig())
    df, feature_cols, _ = _load_logs(data, feature_config=feature_config)
    df, feature_cols, _, _ = _extract_features(
        df, feature_cols, config=feature_config
    )
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
        "symbol,sentiment_timestamp,sentiment_dimension,sentiment_headline_count,sentiment_emb_0,sentiment_emb_1\n",
        "EURUSD,2020-01-01T00:30:00,2,3,0.5,-0.25\n",
    ]
    sentiment.write_text("".join(sentiment_rows))

    feature_config = configure_cache(FeatureConfig())
    df, feature_cols, _ = _load_logs(trades, feature_config=feature_config)
    ns_df = pd.read_csv(sentiment)
    df, feature_cols, _, _ = _extract_features(
        df, feature_cols, news_sentiment=ns_df, config=feature_config
    )
    embed_cols = [c for c in feature_cols if c.startswith("sentiment_emb_")]
    assert embed_cols
    for col in embed_cols:
        assert df[col].notna().all()
    assert "sentiment_headline_count" in feature_cols
    assert (df["sentiment_headline_count"].fillna(0) >= 0).all()


def test_augmentation_adds_rows_and_limits_ranges(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,event_time,symbol\n",
        "0,1.0,100,1.5,2020-01-01T00:00:00,EURUSD\n",
        "1,1.1,110,1.6,2020-01-01T00:01:00,EURUSD\n",
        "0,1.2,120,1.7,2020-01-01T00:02:00,EURUSD\n",
        "1,1.3,130,1.8,2020-01-01T00:03:00,EURUSD\n",
        "0,1.4,140,1.9,2020-01-01T00:04:00,EURUSD\n",
    ]
    data.write_text("".join(rows))
    base_df, _, _ = _load_logs(data, feature_config=configure_cache(FeatureConfig()))
    aug_df, _, _ = _load_logs(
        data, augment_ratio=0.5, feature_config=configure_cache(FeatureConfig())
    )
    assert len(aug_df) > len(base_df)
    min_time = base_df["event_time"].min()
    max_time = base_df["event_time"].max()
    assert pd.to_datetime(aug_df["event_time"]).between(
        min_time - pd.Timedelta("1min"), max_time + pd.Timedelta("1min")
    ).all()
    for col in ["price", "volume", "spread"]:
        lo, hi = base_df[col].min(), base_df[col].max()
        rng = hi - lo
        assert aug_df[col].between(lo - 0.1 * rng, hi + 0.1 * rng).all()


def test_dtw_augmentation_adds_rows_and_limits_ranges(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,event_time,symbol\n",
        "0,1.0,100,1.5,2020-01-01T00:00:00,EURUSD\n",
        "1,1.1,110,1.6,2020-01-01T00:01:00,EURUSD\n",
        "0,1.2,120,1.7,2020-01-01T00:02:00,EURUSD\n",
        "1,1.3,130,1.8,2020-01-01T00:03:00,EURUSD\n",
        "0,1.4,140,1.9,2020-01-01T00:04:00,EURUSD\n",
    ]
    data.write_text("".join(rows))
    base_df, _, _ = _load_logs(data, feature_config=configure_cache(FeatureConfig()))
    aug_df, _, _ = _load_logs(
        data,
        augment_ratio=0.5,
        dtw_augment=True,
        feature_config=configure_cache(FeatureConfig()),
    )
    assert len(aug_df) > len(base_df)
    assert "aug_ratio" in aug_df.columns
    assert aug_df["aug_ratio"].dropna().between(0.0, 1.0).all()
    for col in ["price", "volume", "spread"]:
        lo, hi = base_df[col].min(), base_df[col].max()
        assert aug_df[col].between(lo, hi).all()


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
    feature_config = configure_cache(FeatureConfig())
    df, feature_cols, _ = _load_logs(data, feature_config=feature_config)
    df, feature_cols, _, _ = _extract_features(
        df, feature_cols, config=feature_config
    )
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

