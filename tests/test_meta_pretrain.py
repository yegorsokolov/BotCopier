import json

import json
import numpy as np
import pandas as pd

from scripts.meta_adapt import _logistic_grad, evaluate
from meta.meta_pretrain import save_meta_weights, train_meta_initialisation


def _gen_df(sym: str, w: np.ndarray, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(40, len(w)))
    probs = 1.0 / (1.0 + np.exp(-X @ w))
    y = (probs > 0.5).astype(float)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(len(w))])
    df["label"] = y
    df["symbol"] = sym
    return df


def test_meta_initialisation_speedup(tmp_path):
    base = np.array([1.0, -1.0])
    df_a = _gen_df("A", base + np.array([0.1, -0.1]), 0)
    df_b = _gen_df("B", base + np.array([-0.1, 0.1]), 1)
    df = pd.concat([df_a, df_b], ignore_index=True)
    feat_cols = ["f0", "f1"]
    weights = train_meta_initialisation(
        df, feat_cols, inner_steps=25, inner_lr=0.1, meta_lr=0.5
    )
    model_path = tmp_path / "model.json"
    save_meta_weights(
        weights,
        model_path,
        method="reptile",
        inner_steps=25,
        inner_lr=0.1,
        meta_lr=0.5,
    )
    meta_data = json.loads(model_path.read_text())["meta"]
    loaded = np.array(meta_data["weights"])
    assert np.allclose(loaded, weights)

    df_c = _gen_df("C", base + np.array([0.05, -0.05]), 2)
    Xc = df_c[feat_cols].to_numpy()
    yc = df_c["label"].to_numpy()

    def _run_steps(w_init: np.ndarray, steps: int = 5, lr: float = 0.1) -> float:
        w = w_init.copy()
        for _ in range(steps):
            w -= lr * _logistic_grad(w, Xc, yc)
        return evaluate(w, Xc, yc)

    def _steps_to_acc(w_init: np.ndarray, target: float = 0.9, lr: float = 0.1) -> int:
        w = w_init.copy()
        for step in range(1, 51):
            w -= lr * _logistic_grad(w, Xc, yc)
            if evaluate(w, Xc, yc) >= target:
                return step
        return 50

    random_init = np.zeros_like(loaded)
    steps_rand = _steps_to_acc(random_init)
    steps_meta = _steps_to_acc(loaded.copy())
    assert steps_meta <= steps_rand

    acc_rand = _run_steps(random_init)
    acc_meta = _run_steps(loaded.copy())
    assert acc_meta >= acc_rand
