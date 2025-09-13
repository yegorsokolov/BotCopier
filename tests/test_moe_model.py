import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from botcopier.models.registry import get_model
from botcopier.training.pipeline import train

torch = pytest.importorskip("torch")


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
    _, X, R, y, _ = _make_dataset(tmp_path)
    builder = get_model("moe")
    _, pred_fn = builder(
        X,
        y,
        regime_features=R,
        regime_feature_names=["g0", "g1"],
        epochs=200,
        lr=0.1,
    )
    model = pred_fn.model
    with torch.no_grad():
        gates = torch.softmax(
            model.gating(torch.tensor(R, dtype=torch.float32)), dim=1
        ).cpu().numpy()
    assert np.allclose(gates.sum(axis=1), 1.0)


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
