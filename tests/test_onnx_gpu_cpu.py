import json
import types
from pathlib import Path

import scripts.generate_mql4_from_model as gen


def _run_generate(tmp_path: Path, providers):
    model = {
        "model_id": "onnxgpu",
        "magic": 42,
        "coefficients": [0.1],
        "intercept": 0.0,
        "threshold": 0.5,
        "feature_names": ["hour"],
        "onnx_int8": "model.int8.onnx",
    }
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(model))
    fake_ort = types.SimpleNamespace(get_available_providers=lambda: providers)
    orig = gen.ort
    gen.ort = fake_ort
    out_dir = tmp_path / "out"
    gen.generate(model_file, out_dir)
    gen.ort = orig
    generated = next(out_dir.glob("Generated_onnxgpu_*.mq4"))
    return generated.read_text()


def test_gpu_provider(tmp_path: Path):
    content = _run_generate(tmp_path, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert "UseOnnxGPU = true" in content
    assert "model.int8.onnx" in content


def test_cpu_fallback(tmp_path: Path):
    content = _run_generate(tmp_path, ["CPUExecutionProvider"])
    assert "UseOnnxGPU = false" in content
