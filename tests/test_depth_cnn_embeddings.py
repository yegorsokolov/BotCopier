import json
from pathlib import Path

import numpy as np
import pandas as pd

import botcopier.features.engineering as fe
import botcopier.features.technical as technical


def test_depth_cnn_deterministic(tmp_path: Path) -> None:
    fe.configure_cache(fe.FeatureConfig(enabled_features={"orderbook"}))
    df = pd.DataFrame(
        {
            "bid_depth": [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])],
            "ask_depth": [np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])],
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
        }
    )
    df1, feats1, _, _ = technical._extract_features(df.copy(), [])
    state = technical._DEPTH_CNN_STATE
    assert state is not None
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps({"depth_cnn": state}))

    import importlib

    tech = importlib.import_module("botcopier.features.technical")
    tech._DEPTH_CNN_STATE = None
    saved = json.loads(model_file.read_text())["depth_cnn"]
    df2, feats2, _, _ = technical._extract_features(df.copy(), [], depth_cnn=saved)
    emb_cols = [c for c in feats1 if c.startswith("depth_cnn_")]
    for col in emb_cols:
        assert np.allclose(df1[col], df2[col])
