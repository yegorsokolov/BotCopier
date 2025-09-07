import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts import train_price_gan as tpg
from scripts.train_target_clone import train


def _write_ticks(path: Path) -> None:
    lines = ["bid,ask\n"]
    price = 1.0
    for _ in range(20):
        lines.append(f"{price:.2f},{price + 0.01:.2f}\n")
        price += 0.01
    path.write_text("".join(lines))


def test_transformer_with_synthetic_sequences(tmp_path):
    tick_file = tmp_path / "ticks.csv"
    _write_ticks(tick_file)

    seqs = tpg.load_tick_sequences(tick_file, seq_len=6)
    gen = tpg.train_gan(seqs, epochs=1, latent_dim=4)
    gan_path = tmp_path / "gan.pt"
    tpg.save_model(gen, gan_path)

    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,1.1,2\n"
        "0,1.2,3\n"
        "1,1.3,4\n"
        "0,1.4,5\n"
        "1,1.5,6\n"
    )
    out_dir = tmp_path / "out"
    train(
        data,
        out_dir,
        model_type="transformer",
        window=2,
        epochs=1,
        synthetic_model=gan_path,
        synthetic_frac=1.0,
        synthetic_weight=0.5,
    )

    model = json.loads((out_dir / "model.json").read_text())
    assert "synthetic_metrics" in model
    sm = model["synthetic_metrics"]
    assert sm["synthetic_fraction"] > 0
    assert "real" in sm and "all" in sm
