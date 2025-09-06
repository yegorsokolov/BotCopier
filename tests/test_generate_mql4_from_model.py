import json
import subprocess
import sys

import pytest


def test_generated_features(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"feature_names": ["spread", "hour_sin", "hour_cos"]}))

    template = tmp_path / "StrategyTemplate.mq4"
    # minimal template containing placeholder for insertion
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    subprocess.run(
        [sys.executable, "scripts/generate_mql4_from_model.py", "--model", model, "--template", template],
        check=True,
    )

    content = template.read_text()
    assert "case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread" in content
    assert (
        "case 1: return MathSin(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_sin"
        in content
    )
    assert (
        "case 2: return MathCos(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_cos"
        in content
    )

    data = json.loads(model.read_text())
    assert data["feature_names"] == ["spread", "hour_sin", "hour_cos"]


def test_session_models_inserted(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [],
                "session_models": {
                    "asian": {
                        "coefficients": [1.0],
                        "intercept": 0.1,
                        "threshold": 0.5,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                    }
                },
            }
        )
    )

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __SESSION_MODELS__\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "g_coeffs_asian" in content
    assert "g_threshold_asian" in content
    assert "g_feature_mean_asian" in content
    assert "g_feature_std_asian" in content

    data = json.loads(model.read_text())
    assert "feature_mean" in data["session_models"]["asian"]
    assert "feature_std" in data["session_models"]["asian"]


def test_generation_fails_for_unmapped_feature(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"feature_names": ["unknown"]}))

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                "scripts/generate_mql4_from_model.py",
                "--model",
                model,
                "--template",
                template,
            ],
            check=True,
        )
