"""CLI wrapper for the simplified training pipeline.

This script exists for historical compatibility.  The implementation
moved into the :mod:`botcopier` package and is re-exported here so that
older entry points continue to function.  To preserve the legacy public
API we re-export the training helpers that previously lived directly in
this module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features
from botcopier.models.deep import TabTransformer
from botcopier.training.pipeline import detect_resources, run_optuna, train

__all__ = [
    "train",
    "_load_logs",
    "_extract_features",
    "run_optuna",
    "TabTransformer",
    "detect_resources",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--data-dir", type=Path, help="Path to training data directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory where study outputs and the final model will be written",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to configuration file (default: params.yaml)",
    )
    parser.add_argument(
        "--model-type",
        dest="model_type",
        help="Base model type to use for training (default from configuration)",
    )
    parser.add_argument(
        "--search-model",
        dest="search_models",
        action="append",
        help="Model type to include in the Optuna search space (repeatable)",
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        help="Feature name to enable (repeatable)",
    )
    parser.add_argument(
        "--regime-feature",
        dest="regime_features",
        action="append",
        help="Regime feature name to enable (repeatable)",
    )
    parser.add_argument(
        "--vol-weight",
        dest="vol_weight",
        action="store_true",
        help="Enable volatility weighting when training",
    )
    parser.add_argument(
        "--no-vol-weight",
        dest="vol_weight",
        action="store_false",
        help="Disable volatility weighting",
    )
    parser.set_defaults(vol_weight=None)
    parser.add_argument(
        "--flag",
        dest="flag_options",
        action="append",
        help="Feature flag search in the form name or name=true,false",
    )
    parser.add_argument("--max-drawdown", type=float, default=None)
    parser.add_argument("--var-limit", type=float, default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.model_type is not None:
        overrides["model_type"] = args.model_type
    if args.features:
        overrides["features"] = args.features
    if args.regime_features:
        overrides["regime_features"] = args.regime_features
    if args.vol_weight is not None:
        overrides["vol_weight"] = args.vol_weight

    flag_options: dict[str, list[bool]] = {}
    for raw in args.flag_options or []:
        if "=" in raw:
            name, values = raw.split("=", 1)
            parsed: list[bool] = []
            for token in values.split(","):
                token_norm = token.strip().lower()
                if token_norm in {"1", "true", "yes", "on"}:
                    parsed.append(True)
                elif token_norm in {"0", "false", "no", "off"}:
                    parsed.append(False)
            if parsed:
                flag_options[name.strip()] = parsed
        else:
            flag_options[raw.strip()] = [False, True]

    csv_path = args.out_dir / "hyperparams.csv" if args.out_dir else "hyperparams.csv"
    model_path = args.out_dir / "model.json" if args.out_dir else "model.json"

    run_optuna(
        n_trials=args.trials,
        max_drawdown=args.max_drawdown,
        var_limit=args.var_limit,
        study_name=args.study_name,
        storage=args.storage,
        csv_path=csv_path,
        model_json_path=model_path,
        settings_overrides=overrides or None,
        config_path=args.config,
        model_types=args.search_models,
        feature_flags=flag_options or None,
    )
