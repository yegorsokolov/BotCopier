import json
from pathlib import Path
import sys

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
    assert "double ModelCoefficients[] = {0.1, -0.2};" in content
    assert "double ModelIntercept = 0.05;" in content
    assert "double ModelThreshold = 0.6;" in content
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
    assert "GetSLDistance()" in content
    assert "GetTPDistance()" in content


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


def test_day_of_week_feature(tmp_path: Path):
    model = {
        "model_id": "dow",
        "magic": 888,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["day_of_week"],
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    out_dir = tmp_path / "out"
    generate(model_file, out_dir)

    generated = list(out_dir.glob("Generated_dow_*.mq4"))
    assert len(generated) == 1
    with open(generated[0]) as f:
        content = f.read()
    assert "TimeDayOfWeek(TimeCurrent())" in content


def test_sin_cos_features(tmp_path: Path):
    model = {
        "model_id": "sc",
        "magic": 777,
        "coefficients": [0.1] * 4,
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
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
    assert "MathSin(2*M_PI*TimeHour(TimeCurrent())/24)" in content
    assert "MathCos(2*M_PI*TimeDayOfWeek(TimeCurrent())/7)" in content


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
    assert "MagicNumber = 444" in content


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
    assert "HourlyThresholds" in content
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
        "feature_names": ["sma_H1", "rsi_H1", "macd_H1", "macd_signal_H1"],
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
    assert "PERIOD_H1" in content
    assert "CachedSMA" in content


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
    assert "FeatureMean[]" in content
    assert "FeatureStd[]" in content


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
