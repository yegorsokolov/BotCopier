import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler

from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.training.pipeline import train, _load_logs, _extract_features


def test_scaler_robust_with_outliers(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,1.2,1\n"
        "0,1.3,1\n"
        "1,1.4,2\n"
        "0,1.5,2\n"
        "1,1.6,2\n"
        "0,1.7,3\n"
        "1,1.8,3\n"
        "0,100.0,3\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    params = model["session_models"]["asian"]

    config = configure_cache(FeatureConfig())
    df, features, _ = _load_logs(data, feature_config=config)
    df, features, _, _ = _extract_features(df, features, config=config)
    X = df[features].to_numpy(dtype=float)
    skew = pd.DataFrame(X, columns=features).skew().abs()
    skew_cols = skew[skew > 1.0].index.tolist()
    if skew_cols:
        pt = PowerTransformer(method="yeo-johnson")
        idx = [features.index(c) for c in skew_cols]
        X[:, idx] = pt.fit_transform(X[:, idx])
    clip_min = np.quantile(X, 0.01, axis=0)
    clip_max = np.quantile(X, 0.99, axis=0)
    X_c = np.clip(X, clip_min, clip_max)
    scaler = RobustScaler().fit(X_c)

    assert np.allclose(params["feature_mean"], scaler.center_.tolist())
    assert np.allclose(params["feature_std"], scaler.scale_.tolist())

