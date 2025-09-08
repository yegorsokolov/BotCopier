import json
from pathlib import Path

import numpy as np

from scripts.train_target_clone import _load_logs, _extract_features, train


def test_multi_horizon_training(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = ["label,price,volume,spread,hour,symbol\n"]
    for i in range(30):
        rows.append(f"{i%2},{1.0 + i*0.01},{100+i},{1.5 + 0.01*i},{i%24},EURUSD\n")
    data.write_text("".join(rows))

    df, feature_cols, _ = _load_logs(data)
    df, feature_cols, _, _ = _extract_features(df, feature_cols)
    assert df["label_h5"].notna().all()
    assert df["label_h20"].notna().all()

    out_dir = tmp_path / "out"
    train(data, out_dir)

    model = json.loads((out_dir / "model.json").read_text())
    assert set(model["label_columns"]) >= {"label", "label_h5", "label_h20"}
    params = model["models"]["sgd"]
    coef = np.asarray(params["coefficients"], dtype=float)
    intercept = np.asarray(params["intercept"], dtype=float)
    mean = np.asarray(params["feature_mean"], dtype=float)
    std = np.asarray(params["feature_std"], dtype=float)
    low = np.asarray(params["clip_low"], dtype=float)
    high = np.asarray(params["clip_high"], dtype=float)
    X = df[model["feature_names"]].to_numpy(dtype=float)
    X = np.clip(X, low, high)
    Xs = (X - mean) / std
    logits = Xs @ coef.T + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))
    assert probs.shape[1] == len(model["label_columns"])
