import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features
from botcopier.training.pipeline import train


def test_data_contract(tmp_path: Path) -> None:
    """Run full training pipeline and validate model against contract."""
    schema = yaml.safe_load(Path("schemas/strategy_contract.yaml").read_text())
    expected = [f["name"] for f in schema["features"]]

    # prepare synthetic training logs
    data_file = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "1,1.0,100,1.0,0,EURUSD\n",
        "0,1.1,110,1.1,1,EURUSD\n",
        "1,1.2,120,1.2,2,EURUSD\n",
        "0,1.3,130,1.3,3,EURUSD\n",
    ]
    data_file.write_text("".join(rows))

    # run training pipeline
    out_dir = tmp_path / "out"
    predict_fn = train(data_file, out_dir, n_splits=2, cv_gap=1, param_grid=[{}])

    model = json.loads((out_dir / "model.json").read_text())

    # verify feature ordering and coefficient shape
    assert model["feature_names"] == expected
    assert len(model["coefficients"]) == len(expected)

    # validate feature types using extraction pipeline
    logs, feature_names, _ = _load_logs(data_file)
    df, feature_names, *_ = _extract_features(logs, feature_names)
    for spec in schema["features"]:
        col = df[spec["name"]]
        assert pd.api.types.is_numeric_dtype(col)

    # validate prediction contract
    X = df[expected].fillna(0.0).to_numpy(dtype=float)
    if hasattr(predict_fn, "predict_proba"):
        preds = predict_fn.predict_proba(X)[:, 1]
    else:
        preds = predict_fn(X)
    pred_spec = schema["predictions"][0]
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.shape == (len(df),)
    assert np.isfinite(preds).all()
