import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import scripts.generate_mql4_from_model as gen
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
    train(data, out_dir, model_type="transformer", window=2, epochs=1)
    model = json.loads((out_dir / "model.json").read_text())
    assert model["model_type"] == "transformer"
    weights = model["weights"]
    for key in ["q_weight", "k_weight", "v_weight", "out_weight"]:
        assert key in weights and weights[key]
    assert "teacher_metrics" in model
    assert "distilled" in model and model["distilled"]["coefficients"]

    template = tmp_path / "Strategy.mq4"
    template.write_text(Path("StrategyTemplate.mq4").read_text())
    gen.insert_get_feature(out_dir / "model.json", template)
    content = template.read_text()
    assert "g_q_weight" in content
    assert "seq_len=" in content
    assert "g_coeffs_logreg" in content
