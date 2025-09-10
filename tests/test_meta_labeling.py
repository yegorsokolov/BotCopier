import json
import pandas as pd
from pathlib import Path

from botcopier.data.loading import _load_logs
from botcopier.training.pipeline import train


def test_meta_labels_not_null(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,spread,hour\n",
        "1,1.0,0.05,0\n",
        "0,0.9,0.05,1\n",
        "1,1.1,0.05,2\n",
    ]
    data.write_text("".join(rows))
    df, _, _ = _load_logs(
        data,
        take_profit_mult=1.0,
        stop_loss_mult=1.0,
        hold_period=2,
    )
    assert df["meta_label"].notna().all()


def test_meta_labeling_changes_performance(tmp_path: Path) -> None:
    rows = [
        [1, 1.00, 0.05, 0],
        [1, 0.89, 0.05, 1],
        [1, 1.05, 0.05, 2],
        [1, 1.30, 0.05, 3],
        [1, 1.10, 0.05, 4],
        [1, 0.95, 0.05, 5],
        [1, 0.96, 0.05, 6],
        [1, 0.97, 0.05, 7],
        [1, 1.20, 0.05, 8],
        [1, 1.00, 0.05, 9],
    ]
    raw = pd.DataFrame(rows, columns=["label", "price", "spread", "hour"])
    raw_file = tmp_path / "raw.csv"
    raw.to_csv(raw_file, index=False)
    loaded, _, _ = _load_logs(
        raw_file,
        take_profit_mult=1.0,
        stop_loss_mult=1.0,
        hold_period=2,
    )
    base_file = tmp_path / "base.csv"
    loaded[["label", "spread", "hour"]].to_csv(base_file, index=False)
    meta_file = tmp_path / "meta.csv"
    loaded[["label", "spread", "hour", "meta_label"]].to_csv(
        meta_file, index=False
    )
    out1 = tmp_path / "out1"
    train(base_file, out1)
    acc1 = json.loads((out1 / "model.json").read_text())["cv_accuracy"]
    out2 = tmp_path / "out2"
    train(
        meta_file,
        out2,
        use_meta_label=True,
        take_profit_mult=1.0,
        stop_loss_mult=1.0,
        hold_period=2,
    )
    acc2 = json.loads((out2 / "model.json").read_text())["cv_accuracy"]
    assert acc1 != acc2
