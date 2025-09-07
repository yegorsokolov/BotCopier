import json
from pathlib import Path

from scripts.train_target_clone import _extract_features, _load_logs, train


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
    for col in ["price", "volume", "spread"]:
        for feat in [f"{col}_lag_1", f"{col}_lag_5", f"{col}_diff"]:
            assert feat in model["feature_names"]
    df, feature_cols, _ = _load_logs(data)
    df, _, _ = _extract_features(df, feature_cols)
    for col in ["price", "volume", "spread"]:
        for feat in [f"{col}_lag_1", f"{col}_lag_5", f"{col}_diff"]:
            assert df[feat].notna().all()


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

