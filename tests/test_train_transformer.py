import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_target_clone import train


def test_transformer_weights_and_generation(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,1.1,2\n"
        "0,1.2,3\n"
        "1,1.3,4\n"
        "0,1.4,5\n"
        "1,1.5,6\n"
    )
    out_dir = tmp_path / "out"
    model_obj = train(data, out_dir, model_type="transformer", window=2, epochs=1)
    assert next(model_obj.parameters()).device.type == "cpu"
    if torch.cuda.is_available():
        cuda_dir = tmp_path / "out_cuda"
        cuda_model = train(
            data,
            cuda_dir,
            model_type="transformer",
            window=2,
            epochs=1,
            device="cuda",
        )
        assert next(cuda_model.parameters()).device.type == "cuda"
    model = json.loads((out_dir / "model.json").read_text())
    assert model["model_type"] == "transformer"
    weights = model["weights"]
    for key in ["q_weight", "k_weight", "v_weight", "out_weight", "pos_embed_weight"]:
        assert key in weights and weights[key]
    assert model.get("dropout", 0.0) == 0.0
    assert "teacher_metrics" in model
    assert "distilled" in model and model["distilled"]["coefficients"]



def test_transformer_cli_dropout(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,1.1,2\n"
        "0,1.2,3\n"
        "1,1.3,4\n"
        "0,1.4,5\n"
        "1,1.5,6\n"
    )
    out_dir = tmp_path / "cli_out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_target_clone",
            str(data),
            str(out_dir),
            "--model-type",
            "transformer",
            "--window",
            "2",
            "--epochs",
            "1",
            "--dropout",
            "0.25",
        ],
        check=True,
    )
    model = json.loads((out_dir / "model.json").read_text())
    assert model["dropout"] == 0.25
    assert "pos_embed_weight" in model["weights"]
