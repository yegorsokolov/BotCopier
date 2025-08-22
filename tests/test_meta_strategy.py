import pandas as pd

import json
from pathlib import Path

from scripts.meta_strategy import train_meta_model, select_model
from scripts.generate_mql4_from_model import generate


def test_regime_switch_triggers_model_change():
    df = pd.DataFrame(
        {
            "volatility": [-1.0, -0.8, -0.6, 0.6, 0.8, 1.2],
            "hour": [0, 1, 2, 10, 11, 12],
            "spread": [0.9, 0.85, 0.8, 0.2, 0.15, 0.1],
            "best_model": [0, 0, 0, 1, 1, 1],
        }
    )
    params = train_meta_model(df, ["volatility", "hour", "spread"], label_col="best_model")

    low_regime = {"volatility": -1.0, "hour": 0, "spread": 0.9}
    high_regime = {"volatility": 1.0, "hour": 12, "spread": 0.1}

    assert select_model(params, low_regime) != select_model(params, high_regime)


def test_generate_with_gating(tmp_path: Path):
    df = pd.DataFrame(
        {
            "volatility": [-1.0, -0.5, 0.6, 1.2],
            "hour": [0, 1, 10, 11],
            "spread": [0.9, 0.85, 0.2, 0.1],
            "best_model": [0, 0, 1, 1],
        }
    )
    gating = train_meta_model(df, ["volatility", "hour", "spread"], label_col="best_model")
    gating_file = tmp_path / "gating.json"
    gating_file.write_text(json.dumps(gating))

    m1 = {
        "model_id": "m1",
        "coefficients": [0.1, 0.0, 0.0],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["volatility", "hour", "spread"],
    }
    m2 = {
        "model_id": "m2",
        "coefficients": [-0.1, 0.0, 0.0],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["volatility", "hour", "spread"],
    }
    f1 = tmp_path / "m1.json"
    f2 = tmp_path / "m2.json"
    f1.write_text(json.dumps(m1))
    f2.write_text(json.dumps(m2))

    out_dir = tmp_path / "out"
    generate([f1, f2], out_dir, gating_json=gating_file)
    generated = list(out_dir.glob("Generated_m1_*.mq4"))
    assert generated
    content = generated[0].read_text()
    assert "GatingCoefficients" in content
    assert "GatingIntercepts" in content
