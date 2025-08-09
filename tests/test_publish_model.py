import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.publish_model import publish


def test_publish_model(tmp_path: Path):
    model = {
        "coefficients": [0.1, 0.2],
        "intercept": 0.3,
        "threshold": 0.6,
        "hourly_thresholds": [0.1] * 24,
    }
    model_file = tmp_path / "model.json"
    with open(model_file, "w") as f:
        json.dump(model, f)

    files_dir = tmp_path / "files"
    publish(model_file, files_dir)

    out_file = files_dir / "model.json"
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert data["coefficients"] == [0.1, 0.2]
    assert data["intercept"] == 0.3
    assert data["threshold"] == 0.6
    assert data["hourly_thresholds"] == [0.1] * 24


def test_publish_onnx(tmp_path: Path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy")
    files_dir = tmp_path / "files"
    publish(onnx_file, files_dir)
    assert (files_dir / "model.onnx").exists()

