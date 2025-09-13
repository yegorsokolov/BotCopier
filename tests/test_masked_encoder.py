import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from botcopier.training.pipeline import train
from scripts.pretrain_masked import train as pretrain_encoder


def _make_dataset(path: Path) -> Path:
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
        flip_y=0.2,
    )
    cols = ["spread", "volume", "hour_sin", "hour_cos"]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", y)
    df.to_csv(path, index=False)
    return path


def test_masked_encoder_changes_dim_and_improves_metrics(tmp_path: Path) -> None:
    data = _make_dataset(tmp_path / "data.csv")

    out_base = tmp_path / "base"
    train(data, out_base, n_splits=2, mi_threshold=0.0, cluster_correlation=1.0)
    base_model = json.loads((out_base / "model.json").read_text())
    n_base = len(base_model["feature_names"])
    acc_base = base_model["cv_accuracy"]

    enc_dir = tmp_path / "enc"
    pretrain_encoder(
        data,
        enc_dir,
        latent_dim=3,
        mask_ratio=0.3,
        epochs=400,
        batch_size=16,
    )
    enc_path = enc_dir / "masked_encoder.pt"

    out_mask = tmp_path / "mask"
    train(
        data,
        out_mask,
        pretrain_mask=enc_path,
        n_splits=2,
        mi_threshold=0.0,
        cluster_correlation=1.0,
    )
    mask_model = json.loads((out_mask / "model.json").read_text())
    n_mask = len(mask_model["feature_names"])
    acc_mask = mask_model["cv_accuracy"]

    assert n_mask < n_base
    assert acc_mask >= acc_base
    assert mask_model.get("masked_encoder", {}).get("mask_ratio") == 0.3
