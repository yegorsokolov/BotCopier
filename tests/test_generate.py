import json
from pathlib import Path

from scripts.generate_mql4_from_model import generate


def test_generate_mql4_strategy(tmp_path: Path):
    model = {
        "model_id": "test",
        "coefficients": [0.1, -0.2],
        "intercept": 0.05,
        "feature_names": ["a", "b"],
    }
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(model))
    out_dir = tmp_path / "out"
    generated = generate(model_file, out_dir)
    assert generated.exists()
    content = generated.read_text()
    assert "double predict(double &inputs[])" in content
    assert "z += 0.1 * inputs[0]; // a" in content
    assert "z += -0.2 * inputs[1]; // b" in content
    assert "return 1.0 / (1.0 + MathExp(-z));" in content
