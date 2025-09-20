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
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic_settings import BaseSettings

try:  # ``jsonschema`` is an optional dependency
    import jsonschema
except Exception:  # pragma: no cover - library not available
    jsonschema = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SettingsSnapshot:
    """Frozen snapshot of resolved configuration values."""

    digest: str
    data: Dict[str, Dict[str, Any]]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of the stored configuration data."""

        return json.loads(json.dumps(self.data, sort_keys=True))


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
    regime_features: List[str] = []
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
    controller_max_subset_size: Optional[int] = None
    controller_episode_sample_size: Optional[int] = None
    controller_episode_combination_cap: Optional[int] = None
    controller_baseline_momentum: Optional[float] = 0.9
    meta_weights: Optional[Path] = None

    model_config = {"env_prefix": "TRAIN_", "extra": "forbid"}


class ExecutionConfig(BaseSettings):
    """Runtime execution options."""

    use_gpu: bool = False
    trace: bool = False
    trace_exporter: str = "otlp"
    profile: bool = False

    model_config = {"env_prefix": "EXEC_", "extra": "forbid"}


def _serialise_settings(
    data: DataConfig,
    training: TrainingConfig,
    execution: ExecutionConfig | None,
) -> Dict[str, Dict[str, Any]]:
    def _dump(model: BaseSettings) -> Dict[str, Any]:
        dumped = model.model_dump(mode="python", exclude_none=True)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dumped.items()
        }

    serialised: Dict[str, Dict[str, Any]] = {
        "data": _dump(data),
        "training": _dump(training),
    }
    if execution is not None:
        serialised["execution"] = _dump(execution)
    return serialised


def compute_settings_hash(
    data: DataConfig,
    training: TrainingConfig,
    execution: ExecutionConfig | None = None,
) -> str:
    """Compute a deterministic hash for the resolved configuration values."""

    serialised = _serialise_settings(data, training, execution)
    payload = json.dumps(serialised, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_params(
    data: DataConfig,
    training: TrainingConfig,
    execution: ExecutionConfig | None = None,
    path: Path = Path("params.yaml"),
) -> SettingsSnapshot:
    """Persist resolved configuration values to ``params.yaml``."""

    serialised = _serialise_settings(data, training, execution)
    try:
        existing = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        logger.exception("Failed to read existing parameters from %s", path)
        existing = {}
    existing.update(serialised)
    path.write_text(yaml.safe_dump(existing, sort_keys=False))
    payload = json.dumps(serialised, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return SettingsSnapshot(digest=digest, data=serialised)


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

    def _section(name: str, model: type[BaseSettings]) -> Dict[str, Any]:
        value = raw.get(name, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError(f"{name} section must be a mapping")
        fields = model.model_fields
        allowed = {k: v for k, v in value.items() if k in fields}
        extra = {
            k: v
            for k, v in overrides.items()
            if k in fields and v is not None
        }
        return {**allowed, **extra}

    data_cfg = DataConfig(**_section("data", DataConfig))
    train_cfg = TrainingConfig(**_section("training", TrainingConfig))
    exec_cfg = ExecutionConfig(**_section("execution", ExecutionConfig))
    return data_cfg, train_cfg, exec_cfg


__all__ = [
    "DataConfig",
    "TrainingConfig",
    "ExecutionConfig",
    "load_settings",
    "compute_settings_hash",
    "save_params",
    "SettingsSnapshot",
]
