import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from botcopier.features.technical import _extract_features_impl
from botcopier.features.indicator_discovery import evolve_indicators, evaluate_formula


def test_symbolic_indicators_improve_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    y = a * b
    df = pd.DataFrame({"a": a, "b": b, "y": y})
    model = tmp_path / "model.json"
    model.write_text(json.dumps({}))

    # evolve indicator using a and b to predict y
    evolve_indicators(
        df[["a", "b"]],
        ["a", "b"],
        y,
        model,
        n_components=1,
        generations=5,
        population_size=200,
    )
    data = json.loads(model.read_text())
    formulas = data["symbolic_indicators"]["formulas"]
    assert formulas
    series = evaluate_formula(df, ["a", "b"], formulas[0])
    assert np.isfinite(series).all()

    df_feat, cols, *_ = _extract_features_impl(
        df[["a", "b"]].copy(), ["a", "b"], model_json=model
    )
    assert any(c.startswith("sym_") for c in cols)

    X_base = df[["a", "b"]].to_numpy()
    X_feat = df_feat[cols].to_numpy()
    reg = LinearRegression().fit(X_base, y)
    mse_base = np.mean((reg.predict(X_base) - y) ** 2)
    reg2 = LinearRegression().fit(X_feat, y)
    mse_feat = np.mean((reg2.predict(X_feat) - y) ** 2)
    assert mse_feat < mse_base
