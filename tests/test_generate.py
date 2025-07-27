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
