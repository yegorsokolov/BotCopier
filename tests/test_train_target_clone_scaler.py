import json
from pathlib import Path

from scripts.train_target_clone import train


def test_scaler_stats_present(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,1.2,2\n"
        "0,1.3,9\n"
        "1,1.5,10\n"
        "0,1.4,17\n"
        "1,1.6,18\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for sess in ["asian", "london", "newyork"]:
        params = model["session_models"][sess]
        assert "feature_mean" in params
        assert "feature_std" in params
