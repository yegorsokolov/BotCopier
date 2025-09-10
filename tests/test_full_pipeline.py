import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation import evaluate


def test_full_pipeline(tmp_path: Path) -> None:
    # Generate synthetic training logs
    data_file = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "1,1.0,100,1.0,0,EURUSD\n",
        "0,1.1,110,1.1,1,EURUSD\n",
        "1,1.2,120,1.2,2,EURUSD\n",
        "0,1.3,130,1.3,3,EURUSD\n",
    ]
    data_file.write_text("".join(rows))

    # Run training script
    out_dir = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "botcopier.cli",
            "train",
            str(data_file),
            str(out_dir),
        ],
        check=True,
        env=env,
    )

    model_path = out_dir / "model.json"
    assert model_path.exists()
    model = json.loads(model_path.read_text())
    for field in ("coefficients", "intercept", "feature_names"):
        assert field in model

    # Create predictions and actual logs
    pred_file = tmp_path / "preds.csv"
    with open(pred_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots", "probability"])
        writer.writerow(["2024.01.01 00:00:00", "EURUSD", "buy", "0.1", "0.9"])

    actual_file = tmp_path / "actual.csv"
    with open(actual_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["event_time", "action", "ticket", "symbol", "order_type", "lots", "profit"])
        writer.writerow(["2024.01.01 00:00:05", "OPEN", "1", "EURUSD", "0", "0.1", "0"])
        writer.writerow(["2024.01.01 00:01:00", "CLOSE", "1", "EURUSD", "0", "0.1", "10"])

    metrics = evaluate(pred_file, actual_file, window=60, model_json=model_path)
    for field in ("accuracy", "precision", "recall"):
        assert field in metrics

    metrics_file = out_dir / "metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in ("accuracy", "precision", "recall"):
            writer.writerow({"metric": key, "value": metrics[key]})

    assert metrics_file.exists()
    with open(metrics_file) as f:
        content = f.read()
    assert "accuracy" in content and "precision" in content and "recall" in content
