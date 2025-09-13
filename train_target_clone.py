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

from pathlib import Path
import json
import os
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

def _trial_logger(csv_path: Path) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
    """Return a callback that appends trial information to ``csv_path``.

    Each row contains the trial number, suggested parameters, the objective
    value and the random seed stored in ``trial.user_attrs``.
    """

    def _callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        row = {"trial": trial.number, **trial.params, "seed": trial.user_attrs.get("seed"), "value": trial.value}
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    return _callback


def _objective(trial: optuna.trial.Trial) -> float:
    """A tiny deterministic objective function.

    The trial suggests a seed and a single floating point parameter ``x``.  The
    seed is stored in ``trial.user_attrs`` and used to create a reproducible
    noise term so the optimisation has something to minimise.
    """

    seed = trial.suggest_int("seed", 0, 9999)
    trial.set_user_attr("seed", seed)
    x = trial.suggest_float("x", -10.0, 10.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal()
    return (x - 2) ** 2 + noise


def run_optuna(
    n_trials: int = 10,
    csv_path: Path | str = "hyperparams.csv",
    model_json_path: Path | str = "model.json",
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
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, callbacks=[_trial_logger(csv_path)])

    best = study.best_trial
    relative_csv = os.path.relpath(csv_path, model_json_path.parent)
    model_data = {
        "metadata": {
            "hyperparam_log": relative_csv,
            "best_trial": {"number": best.number, "value": best.value},
        }
    }
    model_json_path.write_text(json.dumps(model_data))

    return study


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    run_optuna()
