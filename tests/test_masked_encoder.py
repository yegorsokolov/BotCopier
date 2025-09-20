import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from botcopier.training.pipeline import train
from scripts.pretrain_masked import train as pretrain_encoder

FEATURE_COLS = ["spread", "volume", "hour_sin", "hour_cos"]


def _make_dataset(path: Path) -> Path:
    X, y = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
        flip_y=0.2,
    )
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, "label", y)
    df.to_csv(path, index=False)
    return path


def test_masked_encoder_changes_dim_and_improves_metrics(tmp_path: Path) -> None:
    data = _make_dataset(tmp_path / "data.csv")

    out_base = tmp_path / "base"
    train(
        data,
        out_base,
        n_splits=2,
        mi_threshold=0.0,
        cluster_correlation=1.0,
        force_heavy=True,
        feature_subset=FEATURE_COLS,
    )
    base_model = json.loads((out_base / "model.json").read_text())
    n_base = len(base_model["feature_names"])
    base_metrics = base_model.get("cv_metrics", {})
    acc_base = float(base_metrics.get("accuracy", 0.0))

    enc_dir = tmp_path / "enc"
    pretrain_encoder(
        data,
        enc_dir,
        latent_dim=3,
        mask_ratio=0.3,
        epochs=400,
        batch_size=16,
    )
    enc_path = enc_dir / "masked_encoder.pt"

    out_mask = tmp_path / "mask"
    train(
        data,
        out_mask,
        pretrain_mask=enc_path,
        n_splits=2,
        mi_threshold=0.0,
        cluster_correlation=1.0,
        force_heavy=True,
        feature_subset=FEATURE_COLS,
    )
    mask_model = json.loads((out_mask / "model.json").read_text())
    n_mask = len(mask_model["feature_names"])
    mask_metrics = mask_model.get("cv_metrics", {})
    acc_mask = float(mask_metrics.get("accuracy", 0.0))

    assert n_mask < n_base
    assert acc_mask >= acc_base
    assert mask_model.get("masked_encoder", {}).get("mask_ratio") == 0.3


def test_masked_encoder_round_trip_matches_inference(tmp_path: Path) -> None:
    data = _make_dataset(tmp_path / "data.csv")

    enc_dir = tmp_path / "enc"
    pretrain_encoder(
        data,
        enc_dir,
        latent_dim=3,
        mask_ratio=0.3,
        epochs=200,
        batch_size=16,
    )
    enc_path = enc_dir / "masked_encoder.pt"

    out_dir = tmp_path / "trained"
    train(
        data,
        out_dir,
        pretrain_mask=enc_path,
        n_splits=2,
        mi_threshold=0.0,
        cluster_correlation=1.0,
        force_heavy=True,
        feature_subset=FEATURE_COLS,
    )
    model = json.loads((out_dir / "model.json").read_text())
    encoder_meta = model.get("masked_encoder")
    assert encoder_meta is not None
    weights = np.asarray(encoder_meta.get("weights"), dtype=float)
    assert weights.shape[0] == len(model.get("feature_names", []))
    assert weights.shape[1] == len(encoder_meta.get("input_features", []))
    bias_meta = encoder_meta.get("bias")
    if bias_meta is not None:
        bias = np.asarray(bias_meta, dtype=float)
        assert bias.shape[0] == weights.shape[0]
    else:
        bias = None
    weights_file = encoder_meta.get("weights_file")
    if weights_file:
        w_path = Path(weights_file)
        if not w_path.is_absolute():
            w_path = (out_dir / w_path).resolve()
        assert w_path.exists()

    df = pd.read_csv(data)
    feature_cols = [c for c in df.columns if c != "label"]
    sample = df.iloc[0]
    features_dict = {col: float(sample[col]) for col in feature_cols}
    raw_vector = np.asarray(
        [features_dict[col] for col in encoder_meta.get("input_features", feature_cols)],
        dtype=float,
    )
    manual_encoded = raw_vector @ weights.T
    if bias is not None and bias.shape[0] == manual_encoded.shape[0]:
        manual_encoded = manual_encoded + bias
    manual_encoded = manual_encoded.astype(float)

    coeffs = np.asarray(model.get("coefficients", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))
    clip_low = np.asarray(model.get("clip_low", []), dtype=float)
    clip_high = np.asarray(model.get("clip_high", []), dtype=float)
    center = np.asarray(model.get("feature_mean", []), dtype=float)
    scale = np.asarray(model.get("feature_std", []), dtype=float)
    limit = min(len(manual_encoded), len(coeffs))
    arr_manual = manual_encoded[:limit].copy()
    if clip_low.size >= limit and clip_high.size >= limit:
        arr_manual = np.clip(arr_manual, clip_low[:limit], clip_high[:limit])
    if center.size >= limit and scale.size >= limit:
        safe_scale = np.where(scale[:limit] == 0, 1.0, scale[:limit])
        arr_manual = (arr_manual - center[:limit]) / safe_scale
    score = float(np.dot(coeffs[:limit], arr_manual) + intercept)
    manual_prob = float(1.0 / (1.0 + np.exp(-score)))

    try:
        serve_module = importlib.import_module("botcopier.scripts.serve_model")
    except ModuleNotFoundError:
        serve_module = None
    if serve_module is not None:
        serve = importlib.reload(serve_module)
        serve.MODEL_DIR = out_dir
        serve._configure_model(model)
        raw_sequence = [float(sample[col]) for col in serve.INPUT_COLUMNS]
        serve_features = serve._apply_masked_encoder(raw_sequence)
        if serve.PT is not None and serve.PT_IDX:
            arr = np.asarray([serve_features[j] for j in serve.PT_IDX], dtype=float).reshape(1, -1)
            transformed = serve.PT.transform(arr).ravel().tolist()
            for idx, value in zip(serve.PT_IDX, transformed):
                serve_features[idx] = float(value)
        serve_prob = serve._predict_logistic(serve_features, serve.LINEAR_CONFIG)
    else:
        serve_prob = manual_prob

    replay_module = importlib.import_module("botcopier.scripts.replay_decisions")
    replay = importlib.reload(replay_module)
    replay.MODEL_DIR = out_dir
    replay_prob = replay._predict_logistic(model, features_dict)

    assert serve_prob == pytest.approx(manual_prob, rel=1e-6, abs=1e-6)
    assert replay_prob == pytest.approx(manual_prob, rel=1e-6, abs=1e-6)
