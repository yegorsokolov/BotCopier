import csv
import json
from pathlib import Path

from scripts.train_target_clone import train


def test_train_with_uncertain_file(tmp_path, caplog):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n" "1,1.0,1,1.0\n" "0,-0.5,2,1.1\n"
    )
    uncertain = tmp_path / "uncertain_decisions_labeled.csv"
    with uncertain.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["action", "probability", "threshold", "features", "label"])
        writer.writerow(["buy", "0.55", "0.5", "1.0:0.0:1.0", "1"])
    out_dir = tmp_path / "out"
    with caplog.at_level("INFO"):
        train(data, out_dir, uncertain_file=uncertain, uncertain_weight=3.0)
    model = json.loads((out_dir / "model.json").read_text())
    metrics = model["session_models"]["0"]["metrics"]
    assert "uncertain_accuracy" in metrics
    assert any("uncertain sample metrics" in r.message for r in caplog.records)
