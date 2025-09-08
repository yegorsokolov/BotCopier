import json
from pathlib import Path

from scripts.train_target_clone import train


def test_drift_pruning_changes_features(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,hour,symbol\n",
        "0,0,EURUSD\n",
        "1,1,EURUSD\n",
        "0,2,EURUSD\n",
    ]
    data.write_text("".join(rows))

    out_no = tmp_path / "out_no"
    train(data, out_no)
    model_no = json.loads((out_no / "model.json").read_text())
    assert "hour_sin" in model_no["feature_names"]
    assert "hour_cos" in model_no["feature_names"]

    out_dir = tmp_path / "out"
    drift_scores = {"hour_sin": 0.6}
    train(data, out_dir, drift_scores=drift_scores, drift_threshold=0.5)
    model = json.loads((out_dir / "model.json").read_text())
    assert "hour_sin" not in model["feature_names"]
    assert "hour_cos" in model["feature_names"]
