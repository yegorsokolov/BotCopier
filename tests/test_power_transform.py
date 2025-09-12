import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from botcopier.training.pipeline import train


def test_power_transform_reduces_variance(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=n),
            "spread": rng.exponential(scale=2.0, size=n),
            "hour": np.zeros(n),
        }
    )
    data = tmp_path / "trades_raw.csv"
    df.to_csv(data, index=False)
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    pt = model.get("power_transformer")
    assert pt is not None
    assert "spread" in pt["features"]
    idx = pt["features"].index("spread")
    pt_obj = PowerTransformer(method="yeo-johnson")
    pt_obj.lambdas_ = np.array([pt["lambdas"][idx]])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array([pt["mean"][idx]])
    scaler.scale_ = np.array([pt["scale"][idx]])
    pt_obj._scaler = scaler
    pt_obj.n_features_in_ = 1
    X = df[["spread"]].to_numpy()
    var_before = X.var()
    var_after = pt_obj.transform(X).var()
    assert var_after < var_before

