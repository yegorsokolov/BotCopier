import json
from pathlib import Path

from scripts.train_target_clone import train


def test_price_indicators_persisted(tmp_path: Path) -> None:
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
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for name in [
        "sma",
        "rsi",
        "macd",
        "macd_signal",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "atr",
    ]:
        assert name in model["feature_names"]

