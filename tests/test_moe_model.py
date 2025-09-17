import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
sklearn_linear = pytest.importorskip("sklearn.linear_model")
LogisticRegression = sklearn_linear.LogisticRegression

from botcopier.models.registry import get_model
from botcopier.models.schema import ModelParams
from botcopier.scripts import generate_mql4_from_model as mql4
from botcopier.scripts import replay_decisions as replay_mod
from botcopier.training.pipeline import train


def _make_dataset(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 200
    regime = rng.integers(0, 2, size=n)
    x = rng.normal(size=n)
    y = np.where(regime == 0, x > 0, x < 0).astype(float)
    X = x.reshape(-1, 1)
    R = np.eye(2)[regime]
    data = tmp_path / "trades_raw.csv"
    rows = ["label,feat,g0,g1\n"] + [
        f"{int(y[i])},{X[i,0]},{R[i,0]},{R[i,1]}\n" for i in range(n)
    ]
    data.write_text("".join(rows))
    return data, X, R, y, regime


def test_gating_probabilities_sum_to_one(tmp_path: Path):
    """Gating network should produce a valid probability simplex."""

    _, X, R, y, _ = _make_dataset(tmp_path)
    builder = get_model("moe")
    meta, pred_fn = builder(
        X,
        y,
        regime_features=R,
        regime_feature_names=["g0", "g1"],
        epochs=200,
        lr=0.1,
    )
    arch = meta.get("architecture", {})
    assert arch.get("type") == "MixtureOfExperts"
    assert arch.get("n_experts") == 2
    assert "state_dict" in meta and meta["state_dict"]
    model = pred_fn.model
    with torch.no_grad():
        gates = torch.softmax(
            model.gating(torch.tensor(R, dtype=torch.float32)), dim=1
        ).cpu().numpy()
    assert np.allclose(gates.sum(axis=1), 1.0)

    # Recompute gating probabilities from persisted weights to ensure metadata
    # is consistent with the trained model.
    gw = np.array(meta["regime_gating"]["weights"])
    gb = np.array(meta["regime_gating"]["bias"])
    logits = R @ gw.T + gb
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert np.allclose(probs, gates)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_moe_accuracy_exceeds_individual_experts(tmp_path: Path):
    data, X, R, y, regime = _make_dataset(tmp_path)
    builder = get_model("moe")
    _, pred_fn = builder(
        X,
        y,
        regime_features=R,
        regime_feature_names=["g0", "g1"],
        epochs=200,
        lr=0.1,
    )
    probs = pred_fn(X, R)
    moe_acc = ((probs >= 0.5).astype(int) == y).mean()

    lr0 = LogisticRegression().fit(X[regime == 0], y[regime == 0])
    lr1 = LogisticRegression().fit(X[regime == 1], y[regime == 1])
    acc0 = lr0.score(X, y)
    acc1 = lr1.score(X, y)
    assert moe_acc > max(acc0, acc1)

    out_dir = tmp_path / "out"
    train(data, out_dir, model_type="moe", regime_features=["g0", "g1"])
    saved = json.loads((out_dir / "model.json").read_text())
    assert "regime_gating" in saved and "experts" in saved
    assert "state_dict" in saved and saved["state_dict"]
    arch = saved.get("architecture", {})
    assert arch.get("type") == "MixtureOfExperts"
    assert arch.get("n_experts") == 2


def test_replay_predictions_align_with_trained_model(tmp_path: Path):
    _, X, R, y, _ = _make_dataset(tmp_path)
    builder = get_model("moe")
    meta, pred_fn = builder(
        X,
        y,
        regime_features=R,
        regime_feature_names=["g0", "g1"],
        epochs=200,
        lr=0.1,
    )
    model_dict = {"feature_names": ["feat"], **meta}
    features = {"feat": float(X[0, 0]), "g0": float(R[0, 0]), "g1": float(R[0, 1])}
    expected = float(pred_fn(X[:1], R[:1])[0])
    direct = replay_mod._predict_moe(model_dict.copy(), features)
    assert np.isclose(direct, expected)
    original_flag = replay_mod._HAS_TORCH
    try:
        replay_mod._HAS_TORCH = False
        fallback = replay_mod._predict_moe(model_dict.copy(), features)
    finally:
        replay_mod._HAS_TORCH = original_flag
    assert np.isclose(fallback, expected)


def test_generate_mql4_renders_moe_parameters(tmp_path: Path, monkeypatch):
    params = ModelParams(
        feature_names=["feat"],
        regime_features=["g0", "g1"],
        experts=[
            {"weights": [0.1], "bias": 0.0},
            {"weights": [-0.2], "bias": 0.05},
        ],
        regime_gating={
            "weights": [[0.3, -0.1], [0.2, 0.4]],
            "bias": [0.0, 0.1],
            "feature_names": ["g0", "g1"],
        },
        model_type="moe",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(params.model_dump_json())
    template_path = tmp_path / "StrategyTemplate.mq4"
    template_path.write_text(Path("StrategyTemplate.mq4").read_text())
    monkeypatch.setitem(mql4.FEATURE_MAP, "feat", mql4.FeatureRuntime("0.0"))
    monkeypatch.setitem(mql4.FEATURE_MAP, "g0", mql4.FeatureRuntime("0.0"))
    monkeypatch.setitem(mql4.FEATURE_MAP, "g1", mql4.FeatureRuntime("0.0"))
    mql4.insert_get_feature(model_path, template_path)
    rendered = template_path.read_text()
    assert "bool g_use_moe = true;" in rendered
    assert "double g_moe_expert_weights" in rendered
    assert "double g_moe_gate_weights" in rendered
    assert "GetRegimeFeature" in rendered
