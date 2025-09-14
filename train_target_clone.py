"""Simplified training script used for unit tests.

This module implements a tiny Optuna optimisation loop.  The real project
contains a much more sophisticated training pipeline, but for the purposes of
the unit tests we only need something that exercises a couple of pieces of
infrastructure:

* Hyper-parameter trials are logged to ``hyperparams.csv`` using an Optuna
  callback.
* ``model.json`` records where this log lives via ``metadata.hyperparam_log``
  and stores information about the best trial for later inspection.

The ``TabTransformer`` class and ``detect_resources`` function are stubs that
exist solely so imports from other modules (or historical code) do not fail.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd


class TabTransformer:
    """Stubbed model class.

    The real implementation lives elsewhere.  Only the methods required by old
    imports are present here.
    """

    def __init__(self, *args, **kwargs):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass

    def eval(self):
        pass

    def __call__(self, x):
        return x


def detect_resources() -> dict:
    """Return dummy resource information.

    In the full project this performs hardware introspection.  For tests we
    simply indicate that the training should run in a light-weight mode.
    """

    return {"lite_mode": True}


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------


def _trial_logger(
    csv_path: Path,
) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
    """Return a callback that appends trial information to ``csv_path``.

    Each row contains the trial number, suggested parameters, the objective
    values and the random seed stored in ``trial.user_attrs``.
    """

    def _callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        row = {
            "trial": trial.number,
            **trial.params,
            "seed": trial.user_attrs.get("seed"),
            "profit": trial.values[0],
            "sharpe": trial.values[1],
            "max_drawdown": trial.values[2],
        }
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    return _callback


def _objective_factory(
    max_drawdown: float | None, var_limit: float | None
) -> Callable[[optuna.trial.Trial], tuple[float, float, float]]:
    """Return a multi-objective that includes risk penalties.

    The returned objective yields ``(profit, sharpe, max_drawdown)`` so the
    study can simultaneously optimise for risk and return.
    """

    def _objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        seed = trial.suggest_int("seed", 0, 9999)
        trial.set_user_attr("seed", seed)
        x = trial.suggest_float("x", -10.0, 10.0)
        rng = np.random.default_rng(seed)
        noise = rng.normal()
        risk = abs(x) / 10.0
        trial.set_user_attr("max_drawdown", risk)
        trial.set_user_attr("var_95", risk)
        profit = -((x - 2) ** 2) + noise
        if max_drawdown is not None and risk > max_drawdown:
            profit -= risk - max_drawdown
        if var_limit is not None and risk > var_limit:
            profit -= risk - var_limit
        sharpe = profit / (risk + 1e-6)
        return profit, sharpe, risk

    return _objective


def run_optuna(
    n_trials: int = 10,
    csv_path: Path | str = "hyperparams.csv",
    model_json_path: Path | str = "model.json",
    *,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
) -> optuna.study.Study:
    """Run a small Optuna study and record trial information.

    Parameters
    ----------
    n_trials:
        Number of optimisation trials to run.
    csv_path:
        Location where the hyper-parameter log will be written.
    model_json_path:
        Where to write the resulting ``model.json`` summary.
    """

    csv_path = Path(csv_path)
    model_json_path = Path(model_json_path)

    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"], sampler=sampler
    )
    objective = _objective_factory(max_drawdown, var_limit)
    study.optimize(objective, n_trials=n_trials, callbacks=[_trial_logger(csv_path)])

    def _select_trial() -> optuna.trial.FrozenTrial:
        candidates = [t for t in study.best_trials]
        if max_drawdown is not None:
            candidates = [t for t in candidates if t.values[2] <= max_drawdown]
        if var_limit is not None:
            candidates = [
                t
                for t in candidates
                if t.user_attrs.get("var_95", t.values[2]) <= var_limit
            ]
        if not candidates:
            candidates = list(study.best_trials)
        return max(candidates, key=lambda t: (t.values[0], t.values[1]))

    best = _select_trial()
    relative_csv = os.path.relpath(csv_path, model_json_path.parent)
    risk = best.user_attrs.get("max_drawdown", 0.0)
    model_data = {
        "metadata": {
            "hyperparam_log": relative_csv,
            "selected_trial": {
                "number": best.number,
                "profit": best.values[0],
                "sharpe": best.values[1],
                "max_drawdown": best.values[2],
            },
        },
        "risk_params": {"max_drawdown": max_drawdown, "var_limit": var_limit},
        "risk_metrics": {
            "max_drawdown": risk,
            "var_95": best.user_attrs.get("var_95", risk),
        },
    }
    model_json_path.write_text(json.dumps(model_data))

    return study


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    p = argparse.ArgumentParser()
    p.add_argument("--max-drawdown", type=float, default=None)
    p.add_argument("--var-limit", type=float, default=None)
    p.add_argument("--trials", type=int, default=10)
    args = p.parse_args()
    run_optuna(
        n_trials=args.trials,
        max_drawdown=args.max_drawdown,
        var_limit=args.var_limit,
    )
