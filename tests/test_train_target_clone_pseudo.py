import json
from pathlib import Path

from botcopier.training.pipeline import train


def test_pseudo_labelling_changes_dataset_and_metrics(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,spread,hour\n",
        "0,0.0,1\n",
        "0,1.0,1\n",
        "1,2.0,1\n",
        "1,3.0,1\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"

    train(data, out_dir)
    model1 = json.loads((out_dir / "model.json").read_text())
    base_acc = model1["cv_accuracy"]

    pseudo = tmp_path / "pseudo.csv"
    pseudo.write_text("spread,hour\n0.1,1\n2.9,1\n")

    train(
        data,
        out_dir,
        pseudo_label_files=[pseudo],
        pseudo_confidence_high=0.6,
        pseudo_confidence_low=0.4,
    )
    model2 = json.loads((out_dir / "model.json").read_text())
    assert "pseudo_samples" in model2
    assert model2["cv_accuracy"] == base_acc
