import json
import base64
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler

from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.training.pipeline import (
    train,
    _load_logs,
    _extract_features,
    _clip_train_features,
)


def _train_zero_imputation(df, features):
    X = df[features].to_numpy(dtype=float)
    X = np.nan_to_num(X)
    y = df["label"].astype(int).to_numpy()
    X_c, _, _ = _clip_train_features(X)
    scaler = RobustScaler().fit(X_c)
    clf = SGDClassifier(loss="log_loss", random_state=0)
    clf.partial_fit(scaler.transform(X_c), y, classes=np.array([0, 1]))
    return clf.coef_[0]


def test_imputer_removes_nans_and_affects_coefficients(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n"
        "0,1.0,1\n"
        "1,,1\n"
        "0,1.3,1\n"
        "1,1.4,2\n"
        "0,,2\n"
        "1,1.6,2\n"
        "0,1.7,3\n"
        "1,1.8,3\n"
        "0,100.0,3\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    imputer = pickle.loads(base64.b64decode(model["imputer"]))

    config = configure_cache(FeatureConfig())
    df_raw, features_raw, _ = _load_logs(data, feature_config=config)
    df_raw, _, _, _ = _extract_features(df_raw, features_raw, config=config)
    for col in model["feature_names"]:
        if col not in df_raw.columns:
            df_raw[col] = 0.0
    X = df_raw[model["feature_names"]].to_numpy(dtype=float)
    X_imp = imputer.transform(X)
    assert not np.isnan(X_imp).any()

    baseline_coef = _train_zero_imputation(df_raw.copy(), model["feature_names"])
    session = next(iter(model["session_models"]))
    params = model["session_models"][session]
    model_coef = np.array(params["coefficients"], dtype=float)
    assert not np.allclose(model_coef, baseline_coef)
