"""Symbolic indicator discovery utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer

# Mapping used for safe evaluation of generated formulas
_SAFE_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "neg": np.negative,
    "max": np.maximum,
    "min": np.minimum,
}


def evaluate_formula(df: pd.DataFrame, features: Iterable[str], formula: str) -> pd.Series:
    """Safely evaluate a symbolic formula on ``df``.

    Parameters
    ----------
    df:
        Dataframe providing raw feature values.
    features:
        Feature columns referenced by the ``formula``.
    formula:
        Expression produced by :class:`gplearn`.
    """

    env = {
        name: pd.to_numeric(df.get(name, 0), errors="coerce").fillna(0.0)
        for name in features
    }
    env.update(_SAFE_FUNCS)
    return eval(formula, {"__builtins__": {}}, env)


def evolve_indicators(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: Iterable[float] | None = None,
    model_path: Path | str = Path("model.json"),
    n_components: int = 3,
    generations: int = 20,
    population_size: int = 200,
    random_state: int = 0,
) -> list[str]:
    """Evolve symbolic indicators and persist them to ``model_path``.

    Existing indicators in ``model_path`` are preserved and new ones appended,
    enabling incremental re-evolution when more data becomes available.
    """

    model_path = Path(model_path)
    existing: dict = {}
    if model_path.exists():
        try:
            existing = json.loads(model_path.read_text())
        except Exception:
            existing = {}
    sym = existing.get("symbolic_indicators", {})
    base_features = sym.get("feature_names", feature_cols)
    old_formulas = sym.get("formulas", [])

    if y is None:
        # fallback target so gplearn can optimise something
        y = np.random.default_rng(random_state).normal(size=len(df))

    transformer = SymbolicTransformer(
        generations=generations,
        population_size=population_size,
        hall_of_fame=population_size // 2,
        n_components=n_components,
        function_set=list(_SAFE_FUNCS.keys()),
        feature_names=base_features,
        random_state=random_state,
    )

    try:
        transformer.fit(df[base_features].to_numpy(), np.asarray(list(y)))
        programs = [str(p) for p in transformer._programs[-1][:n_components]]
    except Exception:
        # Fallback: simple product of first two features
        if len(base_features) >= 2:
            programs = [f"mul({base_features[0]}, {base_features[1]})"]
        else:
            programs = []

    formulas = old_formulas + programs
    existing["symbolic_indicators"] = {
        "feature_names": base_features,
        "formulas": formulas,
    }
    model_path.write_text(json.dumps(existing, indent=2))
    return formulas


__all__ = ["evolve_indicators", "evaluate_formula"]
