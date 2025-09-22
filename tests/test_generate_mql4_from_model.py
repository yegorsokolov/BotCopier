import json
import subprocess
import sys
from pathlib import Path

import pytest


_SAMPLE_MODELS = Path(__file__).resolve().parent / "sample_models"


def _copy_template(tmp_path: Path) -> Path:
    template_src = Path(__file__).resolve().parents[1] / "StrategyTemplate.mq4"
    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text(template_src.read_text())
    return template


@pytest.fixture
def lag_diff_inputs(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text((_SAMPLE_MODELS / "lag_diff_model.json").read_text())
    template = _copy_template(tmp_path)
    return model_path, template


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
    # minimal template containing placeholders for insertion
    template.write_text("#property strict\n\n// __INDICATOR_FUNCTIONS__\n// __GET_FEATURE__\n")

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
    assert "case 0: return MarketSeries(MODE_SPREAD, 0); // spread" in content
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
    assert "double MarketSeries(int mode, int shift)" in content

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


def test_lag_diff_features_resolve_runtime_expressions(lag_diff_inputs):
    model_path, template = lag_diff_inputs

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model_path,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "case 0: return iClose(Symbol(), PERIOD_CURRENT, 0); // price" in content
    assert "case 1: return iClose(Symbol(), PERIOD_CURRENT, 1); // price_lag_1" in content
    assert "case 2: return iClose(Symbol(), PERIOD_CURRENT, 5); // price_lag_5" in content
    assert (
        "case 3: return iClose(Symbol(), PERIOD_CURRENT, 0) - iClose(Symbol(), PERIOD_CURRENT, 1); // price_diff"
        in content
    )
    assert "case 4: return iVolume(Symbol(), PERIOD_CURRENT, 0); // volume" in content
    assert "case 5: return iVolume(Symbol(), PERIOD_CURRENT, 1); // volume_lag_1" in content
    assert (
        "case 6: return iVolume(Symbol(), PERIOD_CURRENT, 0) - iVolume(Symbol(), PERIOD_CURRENT, 1); // volume_diff"
        in content
    )
    assert "case 7: return MarketSeries(MODE_SPREAD, 0); // spread" in content
    assert "case 8: return MarketSeries(MODE_SPREAD, 1); // spread_lag_1" in content
    assert "case 9: return MarketSeries(MODE_SPREAD, 5); // spread_lag_5" in content
    assert (
        "case 10: return MarketSeries(MODE_SPREAD, 0) - MarketSeries(MODE_SPREAD, 1); // spread_diff"
        in content
    )
    assert (
        "case 11: return MarketSeries(MODE_SPREAD, 0) * MarketSeries(MODE_SPREAD, 1); // spread*spread_lag_1"
        in content
    )
    assert (
        "case 12: return MarketSeries(MODE_SPREAD, 0) * (MarketSeries(MODE_SPREAD, 0) - MarketSeries(MODE_SPREAD, 1)); // spread*spread_diff"
        in content
    )
    assert "double MarketSeries(int mode, int shift)" in content


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
    assert "No runtime expressions for features" in exc.value.stderr
    assert "unknown" in exc.value.stderr


def test_logistic_sample_model_generates_indicator_helpers(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text((_SAMPLE_MODELS / "logistic_model.json").read_text())
    template = _copy_template(tmp_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model_path,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "FftMagnitude(0)" in content
    assert "FftPhase(1)" in content
    assert "DepthMicroprice()" in content
    assert "DepthCnnEmbedding(0)" in content
    assert "// __INDICATOR_FUNCTIONS__" not in content


def test_transformer_sample_enables_transformer_block(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text((_SAMPLE_MODELS / "transformer_model.json").read_text())
    template = _copy_template(tmp_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model_path,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "bool g_use_transformer = true;" in content
    assert "int g_transformer_window = 16;" in content
    assert "double g_embed_weight[]" in content
    assert "FftMagnitude(0)" in content


def test_gnn_model_renders_symbol_embeddings(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text((_SAMPLE_MODELS / "gnn_model.json").read_text())
    template = _copy_template(tmp_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model_path,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "double g_emb_EURUSD[]" in content
    assert "double g_emb_USDJPY[]" in content
    assert "case 1: return GraphEmbedding(0); // graph_emb0" in content
    assert "GetRegimeFeature" in content and "GraphEmbedding(0)" in content
    assert "// __SYMBOL_EMBEDDINGS_START__" not in content
