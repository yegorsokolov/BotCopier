import json
from pathlib import Path

from scripts.generate_mql4_from_model import generate


def _select(gating, x):
    coeff = gating["coefficients"]
    intercepts = gating["intercepts"]
    best = 0
    best_z = -1e9
    for idx, (row, b) in enumerate(zip(coeff, intercepts)):
        z = b
        for c in row:
            z += c * x
        if z > best_z:
            best_z = z
            best = idx
    return best


def test_regime_gating_selects_models(tmp_path: Path):
    model = {
        "model_id": "regime",
        "feature_names": ["hour"],
        "meta_model": {
            "feature_names": ["hour"],
            "coefficients": [[1.0], [-1.0]],
            "intercepts": [0.0, 0.0],
        },
        "regime_models": [
            {"regime": 0, "coefficients": [1.0], "intercept": 0.0, "feature_names": ["hour"]},
            {"regime": 1, "coefficients": [-1.0], "intercept": 0.0, "feature_names": ["hour"]},
        ],
        "threshold": 0.5,
    }
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(model))
    out_dir = tmp_path / "out"
    generate(model_file, out_dir)
    gating = model["meta_model"]
    assert _select(gating, 1.0) != _select(gating, -1.0)
