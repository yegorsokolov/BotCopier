import json
from pathlib import Path

import pandas as pd
import torch

from scripts.pretrain_contrastive import train as pretrain_encoder
from botcopier.features.engineering import _extract_features, train
from scripts.replay_decisions import _recompute


def _write_ticks(dir_path: Path, n: int = 50) -> None:
    """Create a minimal tick CSV for pretraining."""
    file = dir_path / "ticks_0.csv"
    with file.open("w") as f:
        f.write("bid\n")
        for i in range(n):
            f.write(f"{float(i)}\n")


def test_contrastive_encoder_flow(tmp_path: Path):
    tick_dir = tmp_path / "ticks"
    tick_dir.mkdir()
    _write_ticks(tick_dir)
    enc_dir = tmp_path / "enc"
    pretrain_encoder(tick_dir, enc_dir, window=3, dim=2, epochs=1, batch=8)
    enc_path = enc_dir / "encoder.pt"
    state = torch.load(enc_path, map_location="cpu")
    window = int(state["window"])
    dim = int(state["dim"])
    # verify feature extraction
    df = pd.DataFrame({f"tick_{i}": [float(i)] for i in range(window)})
    df2, feats, _, _ = _extract_features(df.copy(), [], tick_encoder=enc_path)
    for i in range(dim):
        assert f"enc_{i}" in feats
    # prepare training data
    lines = ["label," + ",".join(f"tick_{i}" for i in range(window)) + "\n"]
    for j in range(10):
        ticks = ",".join(str(float(j + i)) for i in range(window))
        lines.append(f"{j%2},{ticks}\n")
    data_file = tmp_path / "trades_raw.csv"
    data_file.write_text("".join(lines))
    out_dir = tmp_path / "out"
    train(data_file, out_dir, tick_encoder=enc_path, mi_threshold=0.0)
    trained = json.loads((out_dir / "model.json").read_text())
    assert "encoder" in trained
    assert trained["encoder"]["dim"] == dim
    # inference transformation using minimal model
    model = {
        "feature_names": [f"enc_{i}" for i in range(dim)],
        "coefficients": [0.0] * dim,
        "intercept": 0.0,
        "encoder": trained["encoder"],
    }
    row = {f"tick_{i}": float(i) for i in range(window)}
    row.update({"probability": 0.0, "profit": 0.0, "decision_id": 1})
    log_df = pd.DataFrame([row])
    _recompute(log_df, model, 0.5, out_dir)
    for i in range(dim):
        assert f"enc_{i}" in log_df.columns
