import json

from scripts.train_target_clone import train


def test_half_life_recorded(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,event_time,price\n"
        "0,2024-01-01,1.0\n"
        "1,2024-01-03,1.1\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir, half_life_days=1.0)
    model = json.loads((out_dir / "model.json").read_text())
    assert model["half_life_days"] == 1.0
