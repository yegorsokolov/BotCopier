import json
import subprocess
import sys

import pytest


def test_generated_features(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [
                    "spread",
                    "slippage",
                    "equity",
                    "margin_level",
                    "volume",
                    "hour_sin",
                    "hour_cos",
                    "month_sin",
                    "month_cos",
                    "dom_sin",
                    "dom_cos",
                ]
            }
        )
    )

    template = tmp_path / "StrategyTemplate.mq4"
    # minimal template containing placeholder for insertion
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

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
    assert "case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread" in content
    assert "case 1: return OrderSlippage(); // slippage" in content
    assert "case 2: return AccountEquity(); // equity" in content
    assert "case 3: return AccountMarginLevel(); // margin_level" in content
    assert "case 4: return iVolume(Symbol(), PERIOD_CURRENT, 0); // volume" in content
    assert (
        "case 5: return MathSin(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_sin"
        in content
    )
    assert (
        "case 6: return MathCos(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_cos"
        in content
    )
    assert (
        "case 7: return MathSin((TimeMonth(TimeCurrent())-1)*2*MathPi()/12); // month_sin"
        in content
    )
    assert (
        "case 8: return MathCos((TimeMonth(TimeCurrent())-1)*2*MathPi()/12); // month_cos"
        in content
    )
    assert (
        "case 9: return MathSin((TimeDay(TimeCurrent())-1)*2*MathPi()/31); // dom_sin"
        in content
    )
    assert (
        "case 10: return MathCos((TimeDay(TimeCurrent())-1)*2*MathPi()/31); // dom_cos"
        in content
    )

    data = json.loads(model.read_text())
    assert data["feature_names"] == [
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
    ]


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
                            "conformal_lower": 0.2,
                            "conformal_upper": 0.8,
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
    assert "g_conformal_lower_asian" in content
    assert "g_conformal_upper_asian" in content

    data = json.loads(model.read_text())
    assert "feature_mean" in data["session_models"]["asian"]
    assert "feature_std" in data["session_models"]["asian"]
    assert "conformal_lower" in data["session_models"]["asian"]
    assert "conformal_upper" in data["session_models"]["asian"]


def test_generation_fails_on_unmapped_feature(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"feature_names": ["unknown"]}))

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    with pytest.raises(subprocess.CalledProcessError) as exc:
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
            capture_output=True,
            text=True,
        )
    assert "Update StrategyTemplate.mq4" in exc.value.stderr
