import json
from pathlib import Path

import pandas as pd

import sys
import types

gplearn_mod = types.ModuleType("gplearn")
gplearn_genetic = types.ModuleType("gplearn.genetic")
gplearn_genetic.SymbolicTransformer = object
sys.modules.setdefault("gplearn", gplearn_mod)
sys.modules.setdefault("gplearn.genetic", gplearn_genetic)

import botcopier.features.engineering as fe
from botcopier.features.technical import _extract_features_impl
from botcopier.training.pipeline import train


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "close": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
            "volume": [100, 110, 120, 130, 140, 150],
            "price": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
            "symbol": ["EURUSD"] * 6,
            "event_time": pd.date_range("2020-01-01", periods=6, freq="H"),
            "label": [0, 1, 0, 1, 0, 1],
        }
    )


def test_extract_features_deterministic_n_jobs() -> None:
    df = _sample_df()
    feature_names: list[str] = []
    config = fe.configure_cache(fe.FeatureConfig())
    df1, fn1, emb1, gnn1 = _extract_features_impl(
        df.copy(), feature_names.copy(), n_jobs=1, config=config
    )
    df2, fn2, emb2, gnn2 = _extract_features_impl(
        df.copy(), feature_names.copy(), n_jobs=2, config=config
    )
    pd.testing.assert_frame_equal(df1, df2)
    assert fn1 == fn2
    assert emb1 == emb2
    assert gnn1 == gnn2


def test_engineering_wrapper_respects_n_jobs() -> None:
    df = _sample_df()
    feature_names: list[str] = []
    config1 = fe.configure_cache(fe.FeatureConfig(n_jobs=1))
    df1, fn1, emb1, gnn1 = fe._extract_features(
        df.copy(), feature_names.copy(), config=config1
    )
    config2 = fe.configure_cache(fe.FeatureConfig(n_jobs=2))
    df2, fn2, emb2, gnn2 = fe._extract_features(
        df.copy(), feature_names.copy(), config=config2
    )
    pd.testing.assert_frame_equal(df1, df2)
    assert fn1 == fn2
    assert emb1 == emb2
    assert gnn1 == gnn2


def test_train_deterministic_n_jobs(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    df = _sample_df()
    df.to_csv(data, index=False)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    train(data, out1, n_jobs=1, n_splits=2)
    train(data, out2, n_jobs=2, n_splits=2)
    model1 = json.loads((out1 / "model.json").read_text())
    model2 = json.loads((out2 / "model.json").read_text())
    assert model1 == model2
