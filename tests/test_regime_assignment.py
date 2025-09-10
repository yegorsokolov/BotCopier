import json
import logging
from pathlib import Path

import botcopier.features.engineering as fe
from botcopier.data.loading import _load_logs
from botcopier.features.engineering import FeatureConfig, clear_cache, configure_cache
from botcopier.features.technical import _extract_features
from botcopier.training.pipeline import train


def _write_regime_model(path: Path) -> None:
    model = {
        "feature_names": ["price"],
        "mean": [5.0],
        "std": [2.0],
        "centers": [[-2.0], [2.0]],
    }
    path.write_text(json.dumps(model))


def test_regime_labels_assigned(tmp_path: Path, caplog) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.0,0,EURUSD\n",
        "1,9.0,110,1.0,1,EURUSD\n",
    ]
    data.write_text("".join(rows))
    regime_path = tmp_path / "regime_model.json"
    _write_regime_model(regime_path)
    cache_dir = tmp_path / "cache"
    configure_cache(FeatureConfig(cache_dir=cache_dir))
    clear_cache()
    df, feature_cols, _ = _load_logs(data)
    with caplog.at_level(logging.INFO):
        df, feature_cols, _, _ = _extract_features(
            df, feature_cols, regime_model=regime_path
        )
        _extract_features(df, feature_cols, regime_model=regime_path)
    assert "cache hit for _extract_features" in caplog.text
    assert df["regime"].tolist() == [0, 1]
    assert df["regime_0"].tolist() == [1.0, 0.0]
    assert df["regime_1"].tolist() == [0.0, 1.0]
    assert "regime_0" in feature_cols and "regime_1" in feature_cols


def test_per_regime_training(tmp_path: Path, caplog) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.0,0,EURUSD\n",
        "1,2.0,110,1.0,1,EURUSD\n",
        "0,9.0,120,1.0,2,EURUSD\n",
        "1,8.5,130,1.0,3,EURUSD\n",
    ]
    data.write_text("".join(rows))
    regime_path = tmp_path / "regime_model.json"
    _write_regime_model(regime_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    configure_cache(FeatureConfig(cache_dir=cache_dir))
    clear_cache()
    with caplog.at_level(logging.INFO):
        df, feature_cols, _ = _load_logs(data)
        _extract_features(df, feature_cols, regime_model=regime_path)
        _extract_features(df, feature_cols, regime_model=regime_path)
    assert "cache hit for _extract_features" in caplog.text
    train(data, out_dir, regime_model=regime_path, per_regime=True, cache_dir=cache_dir)
    model = json.loads((out_dir / "model.json").read_text())
    sess = model["session_models"]
    assert set(sess.keys()) == {"regime_0", "regime_1"}
    assert sess["regime_0"]["n_samples"] == 2
    assert sess["regime_1"]["n_samples"] == 2
    gating = model.get("regime_gating")
    assert gating and sorted(gating.get("classes", [])) == [0, 1]
