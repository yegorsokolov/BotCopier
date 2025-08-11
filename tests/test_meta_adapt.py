import numpy as np
from pathlib import Path

from scripts.meta_adapt import ReptileMetaLearner, evaluate, log_adaptation


def _make_session(w: np.ndarray, n: int = 100):
    X = np.random.randn(n, len(w))
    y = (1 / (1 + np.exp(-(X @ w))) > 0.5).astype(float)
    return X, y


def test_adaptation_improves_accuracy(tmp_path: Path):
    # two regimes with opposite weights
    w_a = np.array([1.0, -1.0])
    w_b = -w_a

    Xa, ya = _make_session(w_a)
    Xb_train, yb_train = _make_session(w_b)
    Xb_val, yb_val = _make_session(w_b)

    meta = ReptileMetaLearner(dim=2)
    # meta-train only on regime A to force adaptation for regime B
    meta.train([(Xa, ya)], inner_steps=25, inner_lr=0.1, meta_lr=0.5)

    base_acc = evaluate(meta.weights, Xb_val, yb_val)
    new_w = meta.adapt(Xb_train, yb_train, inner_steps=25, inner_lr=0.1)
    adapt_acc = evaluate(new_w, Xb_val, yb_val)

    # ensure weights actually changed and accuracy improved
    assert not np.allclose(meta.weights, new_w)
    assert adapt_acc >= base_acc

    log_file = tmp_path / "adapt.log"
    log_adaptation(meta.weights, new_w, regime_id=1, log_path=str(log_file))
    contents = log_file.read_text().strip().split(",")
    assert contents[1] == "1"
    assert len(contents[-1].split()) == len(new_w)
