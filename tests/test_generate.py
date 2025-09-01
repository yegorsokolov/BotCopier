import json
import re
from pathlib import Path
import sys
import pytest
import numpy as np
from sklearn.feature_extraction import FeatureHasher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_mql4_from_model import generate


def test_generate(tmp_path: Path):
    model = {
        "model_id": "test",
        "magic": 777,
        "coefficients": [0.1, -0.2],
        "intercept": 0.05,
        "threshold": 0.6,
        "feature_names": ["hour", "spread"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_test_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "MagicNumber = 777" in content
    assert "double ModelCoefficients[1][2] = {{0.1, -0.2}};" in content
    assert "double ModelIntercepts[] = {0.05};" in content
    assert "double CalibrationCoef = 1" in content
    assert "double CalibrationIntercept = 0" in content
    assert "double DefaultThreshold = 0.6;" in content
    assert "TimeHour(TimeCurrent())" in content
    assert "MODE_SPREAD" in content


def test_sl_tp_features(tmp_path: Path):
    model = {
        "model_id": "tp_sl",
        "magic": 555,
        "coefficients": [0.1, 0.2],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["sl_dist", "tp_dist"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_tp_sl_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "SLDistance()" in content
    assert "TPDistance()" in content


def test_polynomial_features(tmp_path: Path):
    model = {
        "model_id": "poly",
        "magic": 42,
        "coefficients": [0.1, 0.2, 0.3],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["spread", "lots", "spread*lots"],
    }
    mf = tmp_path / "model.json"
    with open(mf, "w") as f:
        json.dump(model, f)
    out_dir = tmp_path / "out"
    generate(mf, out_dir)
    generated = list(out_dir.glob("Generated_poly_*.mq4"))
    assert generated
    content = generated[0].read_text()
    assert "MarketInfo(SymbolToTrade, MODE_SPREAD) * Lots" in content


def test_generate_ensemble(tmp_path: Path):
    m1 = {
        "model_id": "m1",
        "magic": 123,
        "coefficients": [0.1, -0.2],
        "intercept": 0.05,
        "threshold": 0.5,
        "feature_names": ["hour", "spread"],
    }
    m2 = {
        "model_id": "m2",
        "coefficients": [0.3, 0.4],
        "intercept": -0.1,
        "threshold": 0.5,
        "feature_names": ["spread", "lots"],
    }
    f1 = tmp_path / "m1.json"
    f2 = tmp_path / "m2.json"
    with open(f1, "w") as f:
        json.dump(m1, f)
    with open(f2, "w") as f:
        json.dump(m2, f)
    out_dir = tmp_path / "out"
    generate([f1, f2], out_dir)
    generated = list(out_dir.glob("Generated_m1_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "double ModelCoefficients[2][3]" in content
    assert "double ModelIntercepts[] = {0.05, -0.1};" in content
    assert "Lots" in content
    assert "MODE_SPREAD" in content
    assert "TimeHour(TimeCurrent())" in content


def test_generate_sl_tp_coeffs(tmp_path: Path):
    model = {
        "model_id": "slcoeff",
        "magic": 999,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour"],
        "sl_coefficients": [0.2],
        "sl_intercept": 0.01,
        "tp_coefficients": [0.3],
        "tp_intercept": 0.02,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_slcoeff_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "SLModelCoefficients" in content
    assert "TPModelCoefficients" in content
    assert "GetNewSL(" in content


def test_sin_cos_features(tmp_path: Path):
    model = {
        "model_id": "sc",
        "magic": 777,
        "coefficients": [0.1] * 8,
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "dom_sin",
            "dom_cos",
        ],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_sc_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "HourSin()" in content
    assert "DowCos()" in content
    assert "MonthSin()" in content
    assert "DomCos()" in content


def test_volatility_feature(tmp_path: Path):
    model = {
        "model_id": "vol",
        "magic": 123,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["volatility"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_vol_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "StdDevRecentTicks()" in content


def test_atr_bollinger_features(tmp_path: Path):
    model = {
        "model_id": "ind",
        "magic": 321,
        "coefficients": [0.1] * 4,
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": [
            "atr",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        ],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_ind_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "iATR(SymbolToTrade" in content
    assert "iBands(SymbolToTrade" in content


def test_unknown_feature_raises(tmp_path: Path):
    model = {
        "model_id": "unknown",
        "magic": 111,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["mystery_feature"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="mystery_feature"):
        generate(model_file, out_dir)


def test_news_sentiment_feature(tmp_path: Path):
    model = {
        "model_id": "ns",
        "magic": 111,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["news_sentiment"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    generated = list(out_dir.glob("Generated_ns_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "GetNewsSentiment()" in content


def test_stochastic_adx_features(tmp_path: Path):
    model = {
        "model_id": "stoch",
        "magic": 654,
        "coefficients": [0.1] * 3,
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["stochastic_k", "stochastic_d", "adx"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_stoch_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "iStochastic(SymbolToTrade" in content
    assert "iADX(SymbolToTrade" in content


def test_generate_nn(tmp_path: Path):
    model = {
        "model_id": "nn",
        "magic": 222,
        "hidden_size": 2,
        "nn_weights": [
            [[0.1, 0.2], [0.3, 0.4]],
            [0.0, 0.1],
            [[0.5], [0.6]],
            [0.2],
        ],
        "threshold": 0.5,
        "feature_names": ["hour", "spread"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_nn_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "NNLayer1Weights" in content
    assert "MagicNumber = 222" in content


def test_generate_lstm(tmp_path: Path):
    model = {
        "model_id": "lstm",
        "magic": 333,
        "feature_names": ["hour", "spread"],
        "sequence_length": 2,
        "lstm_weights": [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[0.9, 1.0, 1.1, 1.2]],
            [0.0, 0.1, 0.2, 0.3],
            [[1.3], [1.4]],
            [0.5],
        ],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_lstm_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "LSTMSequenceLength" in content
    assert "MagicNumber = 333" in content


def test_generate_transformer(tmp_path: Path):
    model = {
        "model_id": "trans",
        "magic": 444,
        "feature_names": ["hour", "spread"],
        "sequence_length": 2,
        "transformer_weights": [
            [[[0.1, 0.2]], [[0.3, 0.4]]],
            [[0.0, 0.0]],
            [[[0.5, 0.6]], [[0.7, 0.8]]],
            [[0.0, 0.0]],
            [[[0.9, 1.0]], [[1.1, 1.2]]],
            [[0.0, 0.0]],
            [[[1.3, 1.4]], [[1.5, 1.6]]],
            [0.0, 0.1],
            [[1.7], [1.8]],
            [0.2],
        ],
        "rl_algo": "decision_transformer",
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_trans_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "TransformerDenseWeights" in content
    assert "ComputeDecisionTransformerScore" in content
    assert "MagicNumber = 444" in content
    assert "ModelOnnxFile = \"decision_transformer.onnx\"" in content


def test_generate_hourly_thresholds(tmp_path: Path):
    model = {
        "model_id": "hour", 
        "magic": 111,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "hourly_thresholds": [0.5] * 24,
        "feature_names": ["hour"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_hour_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "ModelThreshold" in content
    assert "GetTradeThreshold()" in content


def test_generate_ratio_feature(tmp_path: Path):
    model = {
        "model_id": "ratio",
        "magic": 999,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["ratio_EURUSD_USDCHF"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_ratio_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert 'iClose("EURUSD", 0, 0) / iClose("USDCHF", 0, 0)' in content


def test_generate_rl_fused(tmp_path: Path):
    model = {
        "model_id": "rl_fused",
        "q_weights": [[0.2, -0.1], [-0.2, 0.1]],
        "q_intercepts": [0.1, -0.1],
        "threshold": 0.5,
        "feature_names": ["hour", "spread"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_rl_fused_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "ModelCoefficients" in content


def test_generate_higher_tf(tmp_path: Path):
    model = {
        "model_id": "hft",
        "magic": 1111,
        "coefficients": [0.1, 0.2, 0.3, 0.4],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["sma_H1", "rsi_H4", "macd_H1", "macd_signal_H4"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_hft_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "TFIdx(PERIOD_H1)" in content
    assert "TFIdx(PERIOD_H4)" in content
    assert "CachedSMA" in content
    assert "CachedRSI" in content


def test_generate_scaling_arrays(tmp_path: Path):
    model = {
        "model_id": "scale",
        "magic": 111,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour"],
        "feature_mean": [12.0],
        "feature_std": [3.0],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_scale_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    m_mean = re.search(r"double FeatureMean\[] = {([^}]*)};", content)
    m_std = re.search(r"double FeatureStd\[] = {([^}]*)};", content)
    assert m_mean and m_std
    assert m_mean.group(1) == "12"
    assert m_std.group(1) == "3"


def test_generate_account_features(tmp_path: Path):
    model = {
        "model_id": "acct",
        "magic": 101,
        "coefficients": [0.1, 0.2],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["equity", "margin_level"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_acct_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "AccountEquity()" in content
    assert "AccountMarginLevel()" in content


def test_slippage_equity_features(tmp_path: Path):
    model = {
        "model_id": "se",
        "magic": 303,
        "coefficients": [0.1, -0.2, 0.3],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["spread", "slippage", "equity"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_se_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    for i, name in enumerate(model["feature_names"]):
        assert f"case {i}:" in content
    assert "MarketInfo(SymbolToTrade, MODE_SPREAD)" in content
    assert "GetSlippage()" in content
    assert "AccountEquity()" in content


def test_generate_lite_mode(tmp_path: Path):
    model = {
        "model_id": "lite",
        "magic": 1,
        "coefficients": [0.1, 0.2, 0.3],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour", "book_bid_vol", "book_ask_vol"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir, lite_mode=True)

    generated = list(out_dir.glob("Generated_lite_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert content.count("BookBidVol()") == 1
    assert content.count("BookAskVol()") == 1


def test_generate_regime_feature(tmp_path: Path):
    model = {
        "model_id": "regime",
        "magic": 202,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["regime"],
        "encoder_weights": [[1.0]],
        "encoder_window": 1,
        "encoder_centers": [[0.0], [1.0]],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_regime_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "EncoderCenters" in content
    assert "GetRegime()" in content


def test_generate_volume_feature(tmp_path: Path):
    model = {
        "model_id": "volfeat",
        "magic": 303,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["volume"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_volfeat_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "iVolume(SymbolToTrade, 0, 0)" in content


def test_generate_orderbook_metrics(tmp_path: Path):
    model = {
        "model_id": "obmet",
        "magic": 55,
        "coefficients": [0.1, 0.2, 0.3],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["book_spread", "bid_ask_ratio", "book_imbalance_roll"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_obmet_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "BookSpread()" in content
    assert "BidAskRatio()" in content
    assert "BookImbalanceRoll()" in content


def test_manage_open_orders_included(tmp_path: Path):
    model = {
        "model_id": "manage",
        "magic": 111,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_manage_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "ManageOpenOrders()" in content


def test_calendar_features(tmp_path: Path):
    model = {
        "model_id": "cal",
        "magic": 111,
        "coefficients": [0.1, 0.2],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["event_flag", "event_impact"],
        "calendar_events": [["2024-01-01T00:30:00", 1.0, 1]],
        "event_window": 60.0,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_cal_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "CalendarFlag()" in content
    assert "CalendarImpact()" in content


def test_on_tick_logistic_inference(tmp_path: Path):
    model = {
        "model_id": "logi",
        "magic": 42,
        "coefficients": [0.1, -0.2],
        "intercept": 0.05,
        "threshold": 0.6,
        "feature_names": ["hour", "spread"],
        "feature_mean": [12.0, 1.5],
        "feature_std": [3.0, 0.5],
        "sl_coefficients": [0.2, 0.3],
        "sl_intercept": 0.01,
        "tp_coefficients": [0.4, 0.5],
        "tp_intercept": 0.02,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_logi_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "double ModelCoefficients[1][2] = {{0.1, -0.2}};" in content
    assert "double FeatureMean[] = {12, 1.5};" in content
    assert "double FeatureStd[] = {3, 0.5};" in content
    assert "double z = ModelIntercepts[m];" in content
    assert "1.0 / (1.0 + MathExp(-z))" in content
    assert "if(prob > thr)" in content
    assert "GetNewSL(true)" in content
    assert "GetNewTP(true)" in content


def test_symbol_embeddings(tmp_path: Path):
    model = {
        "model_id": "emb",
        "magic": 100,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour"],
        "symbol_embeddings": {
            "EURUSD": [0.1, 0.2],
            "USDJPY": [0.3, 0.4],
        },
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    generated = list(out_dir.glob("Generated_emb_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "EURUSD" in content
    assert "SymbolEmbeddings" in content


def test_pruned_features_omitted(tmp_path: Path):
    model = {
        "model_id": "pruned",
        "magic": 321,
        "coefficients": [0.1, -0.2],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour", "spread"],
        "feature_importance": {"hour": 0.3, "spread": 0.0},
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    generated = list(out_dir.glob("Generated_pruned_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    assert "TimeHour(TimeCurrent())" in content
    assert "// spread" not in content


def test_hashed_feature_alignment(tmp_path: Path):
    model = {
        "model_id": "hash",
        "magic": 111,
        "coefficients": [0.1] * 8,
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour", "spread"],
        "hash_size": 8,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    generated = list(out_dir.glob("Generated_hash_*.mq4"))
    assert len(generated) == 1
    content = generated[0].read_text()
    hasher = FeatureHasher(n_features=8, input_type="dict")
    vec_hour = hasher.transform([{"hour": 1.0}]).toarray()[0]
    idx_hour = int(np.flatnonzero(vec_hour)[0])
    vec_spread = hasher.transform([{"spread": 1.0}]).toarray()[0]
    idx_spread = int(np.flatnonzero(vec_spread)[0])
    assert f"case {idx_hour}:" in content
    assert f"case {idx_spread}:" in content
