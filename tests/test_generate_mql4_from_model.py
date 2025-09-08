import json
import subprocess
import sys
from pathlib import Path

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
                    "atr",
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
    assert (
        "case 11: return iATR(Symbol(), PERIOD_CURRENT, 14, 0); // atr" in content
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
        "atr",
    ]


def test_pruned_features_not_emitted(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": ["spread", "slippage"],
                "retained_features": ["slippage"],
            }
        )
    )

    template = tmp_path / "StrategyTemplate.mq4"
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
    assert "OrderSlippage()" in content
    assert "MODE_SPREAD" not in content


def test_models_and_router_inserted(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [],
                "models": {
                    "logreg": {
                        "coefficients": [1.0],
                        "intercept": 0.1,
                        "threshold": 0.5,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "conformal_lower": 0.2,
                        "conformal_upper": 0.8,
                    },
                    "xgboost": {
                        "coefficients": [0.5],
                        "intercept": -0.2,
                        "threshold": 0.6,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "conformal_lower": 0.1,
                        "conformal_upper": 0.9,
                    },
                    "lstm": {
                        "coefficients": [0.3],
                        "intercept": 0.0,
                        "threshold": 0.4,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "conformal_lower": 0.15,
                        "conformal_upper": 0.85,
                    },
                },
                "ensemble_router": {
                    "intercept": [0.0, 0.1, -0.1],
                    "coefficients": [[0.5, -0.2], [0.1, 0.3], [-0.4, 0.2]],
                    "feature_mean": [0.0, 12.0],
                    "feature_std": [1.0, 6.0],
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
    assert "g_coeffs_logreg" in content
    assert "g_coeffs_xgboost" in content
    assert "g_coeffs_lstm" in content
    assert "g_router_intercept" in content
    assert "g_router_coeffs" in content


def test_on_tick_logs_uncertain_reason(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [],
                "models": {
                    "logreg": {
                        "coefficients": [1.0],
                        "intercept": 0.0,
                        "threshold": 0.5,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "conformal_lower": 0.4,
                        "conformal_upper": 0.6,
                    }
                },
            }
        )
    )

    template_src = Path(__file__).resolve().parents[1] / "StrategyTemplate.mq4"
    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text(template_src.read_text())

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
    assert "prob >= g_conformal_lower && prob <= g_conformal_upper" in content
    assert 'decision = "skip"' in content
    assert ",reason=" in content
    assert "LogUncertainDecision" in content

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
