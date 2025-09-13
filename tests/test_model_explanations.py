import json
from pathlib import Path

from botcopier.training.pipeline import train


def test_explanation_report_created(tmp_path: Path) -> None:
    data_file = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "1,1.0,100,1.0,0,EURUSD\n",
        "0,1.1,110,1.1,1,EURUSD\n",
        "1,1.2,120,1.2,2,EURUSD\n",
        "0,1.3,130,1.3,3,EURUSD\n",
    ]
    data_file.write_text("".join(rows))

    out_dir = tmp_path / "out"
    train(data_file, out_dir, n_splits=2, cv_gap=1, param_grid=[{}])

    model_path = out_dir / "model.json"
    model = json.loads(model_path.read_text())
    assert "explanation_report" in model

    report_path = out_dir / model["explanation_report"]
    assert report_path.exists()
    content = report_path.read_text()
    # ensure top features appear in the report
    for feat in model["feature_names"][:2]:
        assert feat in content
