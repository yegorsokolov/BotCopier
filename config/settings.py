"""Pydantic based configuration models and loader.

The repository historically used ad-hoc dictionaries for passing around
configuration options.  This module centralises configuration handling using
``pydantic`` models which provide both type hints and validation.  Configuration
values can be populated from environment variables, ``params.yaml`` files and
command line arguments.

Three configuration sections are exposed:

``DataConfig``
    File system paths and data related options.
``TrainingConfig``
    Hyper parameters and options affecting model training.
``ExecutionConfig``
    Runtime execution toggles such as tracing or GPU usage.

The :func:`load_settings` helper loads ``params.yaml`` if present, validates the
document against a JSON schema (``schemas/settings.schema.json``) and applies any
overrides supplied by a CLI script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic_settings import BaseSettings

try:  # ``jsonschema`` is an optional dependency
    import jsonschema
except Exception:  # pragma: no cover - library not available
    jsonschema = None  # type: ignore


class DataConfig(BaseSettings):
    """Paths and data-related options."""

    log_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    files_dir: Optional[Path] = None
    metrics_file: Optional[Path] = None
    tick_file: Optional[Path] = None
    baseline_file: Optional[Path] = None
    recent_file: Optional[Path] = None
    uncertain_file: Optional[Path] = None
    csv: Optional[Path] = None
    data: Optional[Path] = None
    out: Optional[Path] = None
    pred_file: Optional[Path] = None
    actual_log: Optional[Path] = None
    drift_scores: Optional[Path] = None
    flag_file: Optional[Path] = None
    data_dir: Optional[Path] = None

    model_config = {"env_prefix": "DATA_", "extra": "forbid"}


class TrainingConfig(BaseSettings):
    """Training and evaluation parameters."""

    win_rate_threshold: float = 0.4
    drawdown_threshold: float = 0.2
    drift_threshold: float = 0.2
    drift_method: str = "psi"
    uncertain_weight: float = 2.0
    interval: Optional[float] = None

    batch_size: int = 32
    lr: float = 0.01
    lr_decay: float = 1.0
    flight_host: str = "127.0.0.1"
    flight_port: int = 8815
    flight_path: str = "trades"
    drift_interval: float = 300.0
    metrics_port: int = 8003
    cache_dir: Optional[Path] = None
    model: Path = Path("model.json")
    features: List[str] = []
    label: str = "best_model"
    model_type: str = "logreg"
    window: int = 60
    random_seed: int = 0
    online_model: str = "sgd"
    grad_clip: float = 1.0
    half_life_days: float = 0.0
    vol_weight: bool = False
    eval_hooks: List[str] = []
    hrp_allocation: bool = False
    strategy_search: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    metrics: List[str] = []
    reuse_controller: bool = False
    meta_weights: Optional[Path] = None

    model_config = {"env_prefix": "TRAIN_", "extra": "forbid"}


class ExecutionConfig(BaseSettings):
    """Runtime execution options."""

    use_gpu: bool = False
    trace: bool = False
    trace_exporter: str = "otlp"
    profile: bool = False

    model_config = {"env_prefix": "EXEC_", "extra": "forbid"}


def save_params(
    data: DataConfig,
    training: TrainingConfig,
    execution: ExecutionConfig | None = None,
    path: Path = Path("params.yaml"),
) -> None:
    """Persist resolved configuration values to ``params.yaml``."""
    try:
        existing = yaml.safe_load(path.read_text()) or {}
    except Exception:
        existing = {}
    existing["data"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in data.model_dump().items()
    }
    existing["training"] = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in training.model_dump().items()
    }
    if execution is not None:
        existing["execution"] = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in execution.model_dump().items()
        }
    path.write_text(yaml.safe_dump(existing, sort_keys=False))


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "settings.schema.json"
)


def _validate_schema(data: Dict[str, Any]) -> None:
    """Validate ``data`` against the settings JSON schema if available."""
    if jsonschema is None:  # pragma: no cover - optional dependency
        return
    try:
        schema = json.loads(SCHEMA_PATH.read_text())
    except FileNotFoundError:  # pragma: no cover - schema optional during dev
        return
    jsonschema.validate(data, schema)


def load_settings(
    overrides: Dict[str, Any] | None = None,
    *,
    path: Path = Path("params.yaml"),
) -> Tuple[DataConfig, TrainingConfig, ExecutionConfig]:
    """Load configuration values from ``path`` applying ``overrides``.

    Parameters
    ----------
    overrides:
        Mapping of CLI arguments. Keys that match fields on the config models
        override values loaded from file/environment.
    path:
        Location of the YAML configuration file. Defaults to ``params.yaml`` in
        the current directory.
    """

    try:
        raw: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raw = {}

    _validate_schema(raw)

    overrides = overrides or {}
    data_over = {k: v for k, v in overrides.items() if k in DataConfig.model_fields}
    train_over = {
        k: v for k, v in overrides.items() if k in TrainingConfig.model_fields
    }
    exec_over = {
        k: v for k, v in overrides.items() if k in ExecutionConfig.model_fields
    }

    data_cfg = DataConfig(**{**raw.get("data", {}), **data_over})
    train_cfg = TrainingConfig(**{**raw.get("training", {}), **train_over})
    exec_cfg = ExecutionConfig(**{**raw.get("execution", {}), **exec_over})
    return data_cfg, train_cfg, exec_cfg


__all__ = [
    "DataConfig",
    "TrainingConfig",
    "ExecutionConfig",
    "load_settings",
    "save_params",
]
