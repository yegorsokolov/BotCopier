import json
from pathlib import Path

from scripts.train_target_clone import train


def test_quantile_model_outputs_quantiles(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,profit,hour,spread\n",
        "1,2,1,1.0\n",
        "0,1,2,0.0\n",
        "1,2,3,1.1\n",
        "0,1,4,0.1\n",
        "1,2,5,1.2\n",
        "0,1,6,0.2\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    train(data, out_dir, quantile_model=True)
    model = json.loads((out_dir / "model.json").read_text())
    preds = model.get("quantile_predictions")
    assert preds is not None
    assert set(preds.keys()) >= {"0.05", "0.5", "0.95"}
    assert len(preds["0.5"]) == 6

