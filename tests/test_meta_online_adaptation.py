import json
from pathlib import Path

import numpy as np

from meta import ReptileMetaLearner
from meta.meta_pretrain import save_meta_weights
from scripts.online_trainer import OnlineTrainer


def _make_session(w: np.ndarray, n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(w)))
    y = (1.0 / (1.0 + np.exp(-(X @ w))) > 0.5).astype(int)
    return X, y


def _to_batch(X: np.ndarray, y: np.ndarray):
    batch = []
    for xi, yi in zip(X, y):
        rec = {f"f{j}": float(xi[j]) for j in range(X.shape[1])}
        rec["y"] = int(yi)
        batch.append(rec)
    return batch


def test_meta_initialisation_speeds_adaptation(tmp_path: Path) -> None:
    np.random.seed(0)
    w_a = np.array([1.0, 1.0])
    w_b = np.array([1.0, 0.5])

    Xa, ya = _make_session(w_a, seed=1)
    Xb_train, yb_train = _make_session(w_b, n=40, seed=2)
    Xb_val, yb_val = _make_session(w_b, n=200, seed=3)

    meta = ReptileMetaLearner(dim=2)
    meta.train([(Xa, ya)], inner_steps=25, inner_lr=0.1, meta_lr=0.5)

    model_path = tmp_path / "model.json"
    save_meta_weights(meta.weights, model_path)

    trainer_meta = OnlineTrainer(model_path=model_path, batch_size=40, run_generator=False)
    trainer_meta.update(_to_batch(Xb_train, yb_train))
    preds_meta = trainer_meta.clf.predict(Xb_val)
    acc_meta = (preds_meta == yb_val).mean()

    scratch_path = tmp_path / "scratch.json"
    trainer_base = OnlineTrainer(model_path=scratch_path, batch_size=40, run_generator=False)
    trainer_base.update(_to_batch(Xb_train, yb_train))
    preds_base = trainer_base.clf.predict(Xb_val)
    acc_base = (preds_base == yb_val).mean()

    assert acc_meta >= acc_base

    data = json.loads(model_path.read_text())
    assert "meta" in data and "weights" in data["meta"]
    assert "adaptation_log" in data and len(data["adaptation_log"]) >= 1

