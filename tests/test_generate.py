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
