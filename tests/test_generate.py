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

    out_file = out_dir / "Generated_test.mq4"
    assert out_file.exists()
    with open(out_file) as f:
        content = f.read()
    assert "MagicNumber = 777" in content
    assert "double ModelCoefficients[] = {0.1, -0.2};" in content
    assert "double ModelIntercept = 0.05;" in content
    assert "double ModelThreshold = 0.6;" in content
    assert "TimeHour(TimeCurrent())" in content
    assert "MODE_SPREAD" in content
