"""Discover symbolic indicators using genetic programming and update model.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import typer
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


def _evaluate_formula(
    df: pd.DataFrame, features: Iterable[str], formula: str
) -> pd.Series:
    env = {
        name: pd.to_numeric(df[name], errors="coerce").fillna(0.0) for name in features
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
    """Evolve symbolic indicators from ``feature_cols`` and update ``model.json``.

    If ``model.json`` already contains indicators they are preserved and new ones
    are appended. This enables incremental re-evolution when new data becomes
    available.
    """

    model_path = Path(model_path)
    existing: dict = {}
    if model_path.exists():
        existing = json.loads(model_path.read_text())
    sym = existing.get("symbolic_indicators", {})
    base_features = sym.get("feature_names", feature_cols)
    old_formulas = sym.get("formulas", [])

    if y is None:
        # fall back to a random target so that gplearn can optimise something
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


def main(
    csv_file: Path,
    model: Path = Path("model.json"),
    target: str | None = None,
    n_components: int = 3,
) -> None:
    """CLI entrypoint to evolve indicators from a CSV of features."""

    df = pd.read_csv(csv_file)
    feature_cols = [c for c in df.columns if c != target]
    y = df[target] if target is not None else None
    evolve_indicators(
        df[feature_cols], feature_cols, y, model, n_components=n_components
    )


if __name__ == "__main__":
    typer.run(main)
