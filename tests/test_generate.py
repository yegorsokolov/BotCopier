import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_mql4_from_model import generate


def test_generate(tmp_path: Path):
    model = {"model_id": "test", "magic": 777}
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
