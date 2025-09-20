import json
import numpy as np
import torch

from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.training.pipeline import (
    _load_logs,
    _extract_features,
    _maybe_smote,
    train,
)


def test_smote_balances_classes(tmp_path):
    data = tmp_path / "trades_raw.csv"
    lines = ["label,spread,hour"]
    for i in range(20):
        lines.append(f"0,{1.0 + i*0.01},{i%24}")
    for i in range(2):
        lines.append(f"1,{2.0 + i*0.01},{i%24}")
    data.write_text("\n".join(lines))

    config = configure_cache(FeatureConfig())
    df, features, _ = _load_logs(data, feature_config=config)
    df, features, _, _ = _extract_features(df, features, config=config)
    X = df[features].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    w = np.ones_like(y, dtype=float)

    X_res, y_res, w_res = _maybe_smote(X, y, w, threshold=1.5)
    orig_counts = np.bincount(y)
    res_counts = np.bincount(y_res)
    assert res_counts[0] == res_counts[1]
    assert res_counts[1] > orig_counts[1]
    assert len(w_res) == len(y_res)


def test_focal_loss_changes_coefficients(tmp_path):
    data = tmp_path / "trades_raw.csv"
    lines = ["label,spread,hour"]
    for i in range(30):
        lines.append(f"0,{1.0 + i*0.01},{i%24}")
    for i in range(3):
        lines.append(f"1,{2.0 + i*0.01},{i%24}")
    data.write_text("\n".join(lines))

    torch.manual_seed(0)
    np.random.seed(0)
    out1 = tmp_path / "out1"
    train(data, out1, model_type="transformer", epochs=1, window=4)
    coeff1 = json.loads((out1 / "model.json").read_text())["distilled"]["coefficients"]

    torch.manual_seed(0)
    np.random.seed(0)
    out2 = tmp_path / "out2"
    train(
        data,
        out2,
        model_type="transformer",
        epochs=1,
        window=4,
        focal_gamma=2.0,
    )
    coeff2 = json.loads((out2 / "model.json").read_text())["distilled"]["coefficients"]
    assert not np.allclose(coeff1, coeff2)
