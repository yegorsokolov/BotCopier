import json
from pathlib import Path

from scripts.train_target_clone import _load_logs, _extract_features, train


def _write_regime_model(path: Path) -> None:
    model = {
        "feature_names": ["price"],
        "mean": [5.0],
        "std": [2.0],
        "centers": [[-2.0], [2.0]],
    }
    path.write_text(json.dumps(model))


def test_regime_labels_assigned(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "0,1.0,100,1.0,0,EURUSD\n",
        "1,9.0,110,1.0,1,EURUSD\n",
    ]
    data.write_text("".join(rows))
    regime_path = tmp_path / "regime_model.json"
    _write_regime_model(regime_path)
    df, feature_cols, _ = _load_logs(data)
    df, feature_cols, _, _ = _extract_features(
        df, feature_cols, regime_model=regime_path
    )
    assert df["regime"].tolist() == [0, 1]
    assert df["regime_0"].tolist() == [1.0, 0.0]
    assert df["regime_1"].tolist() == [0.0, 1.0]
    assert "regime_0" in feature_cols and "regime_1" in feature_cols


def test_per_regime_training(tmp_path: Path) -> None:
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
    train(data, out_dir, regime_model=regime_path, per_regime=True)
    model = json.loads((out_dir / "model.json").read_text())
    sess = model["session_models"]
    assert set(sess.keys()) == {"regime_0", "regime_1"}
    assert sess["regime_0"]["n_samples"] == 2
    assert sess["regime_1"]["n_samples"] == 2
    gating = model.get("regime_gating")
    assert gating and sorted(gating.get("classes", [])) == [0, 1]
