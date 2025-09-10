import json
from pathlib import Path

import pytest

from botcopier.training.pipeline import train


def test_cross_validation_metrics_written(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "1,1.0,1,1.0\n"
        "0,-0.5,2,1.1\n"
        "1,0.2,9,1.2\n"
        "0,-0.3,10,1.3\n"
        "1,0.4,17,1.4\n"
        "0,-0.6,18,1.5\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    assert "cv_accuracy" in model and "cv_profit" in model
    assert "conformal_lower" in model and "conformal_upper" in model
    for params in model["session_models"].values():
        assert params["cv_metrics"]
        assert "conformal_lower" in params and "conformal_upper" in params
        for fm in params["cv_metrics"]:
            assert "accuracy" in fm and "profit" in fm


def test_training_fails_when_threshold_unmet(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "0,-1,1,1.0\n"
        "1,-2,2,1.0\n"
        "0,-1,9,1.0\n"
        "1,-2,10,1.0\n"
        "0,-1,17,1.0\n"
        "1,-2,18,1.0\n"
    )
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        train(data, out_dir, min_accuracy=1.1, min_profit=0.1)
