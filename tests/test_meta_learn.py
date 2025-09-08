import json
import numpy as np
import pandas as pd

from scripts.meta_learn import train_meta_initialisation, save_meta_weights
from scripts.meta_adapt import ReptileMetaLearner, evaluate


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
    weights = train_meta_initialisation(df, feat_cols, inner_steps=25, inner_lr=0.1, meta_lr=0.5)
    model_path = tmp_path / "model.json"
    save_meta_weights(weights, model_path)
    loaded = np.array(json.loads(model_path.read_text())["meta_weights"])
    assert np.allclose(loaded, weights)

    df_c = _gen_df("C", base + np.array([0.05, -0.05]), 2)
    Xc = df_c[feat_cols].to_numpy()
    yc = df_c["label"].to_numpy()

    scratch = ReptileMetaLearner(len(feat_cols))
    w_scratch = scratch.adapt(Xc, yc, inner_steps=5, inner_lr=0.1)
    acc_scratch = evaluate(w_scratch, Xc, yc)

    meta_model = ReptileMetaLearner(len(feat_cols), loaded)
    w_meta = meta_model.adapt(Xc, yc, inner_steps=5, inner_lr=0.1)
    acc_meta = evaluate(w_meta, Xc, yc)

    assert acc_meta >= acc_scratch
