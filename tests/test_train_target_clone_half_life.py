import json

from botcopier.training.pipeline import train


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


def test_decay_weighting_changes_coefficients(tmp_path):
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,event_time,price\n",
        "0,2024-01-01,1.0\n",
        "0,2024-01-02,2.0\n",
        "0,2024-01-03,3.0\n",
        "1,2024-01-04,4.0\n",
    ]
    data.write_text("".join(rows))
    out1 = tmp_path / "out1"
    train(data, out1)
    coeffs1 = json.loads((out1 / "model.json").read_text())["session_models"][
        "asian"
    ]["coefficients"]
    out2 = tmp_path / "out2"
    train(data, out2, half_life_days=0.5)
    coeffs2 = json.loads((out2 / "model.json").read_text())["session_models"][
        "asian"
    ]["coefficients"]
    assert any(abs(a - b) > 1e-6 for a, b in zip(coeffs1, coeffs2))
