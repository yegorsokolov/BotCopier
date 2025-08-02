import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.promote_best_models import promote


def _create_model(dir_path: Path, name: str, metric_value: float):
    model_dir = dir_path / name
    model_dir.mkdir()
    model_file = model_dir / f"{name}.json"
    model_file.write_text("{}")
    with open(model_dir / "evaluation.json", "w") as f:
        json.dump({"accuracy": metric_value}, f)
    return model_file


def test_promote_uses_evaluation(tmp_path: Path):
    m1 = _create_model(tmp_path, "model_a", 0.9)
    _create_model(tmp_path, "model_b", 0.8)

    best_dir = tmp_path / "best"
    promote(tmp_path, best_dir, max_models=1, metric="accuracy")

    assert (best_dir / m1.name).exists()
