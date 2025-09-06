import json
from pathlib import Path

from scripts.train_target_clone import train


def test_extra_price_features(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,hour,symbol\n",
        "0,1.0,0,EURUSD\n",
        "1,1.1,1,EURUSD\n",
        "0,1.2,2,EURUSD\n",
        "1,1.3,3,EURUSD\n",
        "0,1.4,4,EURUSD\n",
    ]
    data.write_text("".join(rows))
    out_dir = tmp_path / "out"
    extra = {"GBPUSD": [1.0, 1.05, 1.1, 1.15, 1.2]}
    train(data, out_dir, extra_prices=extra)
    model = json.loads((out_dir / "model.json").read_text())
    assert "corr_EURUSD_GBPUSD" in model["feature_names"]
    assert "ratio_EURUSD_GBPUSD" in model["feature_names"]
