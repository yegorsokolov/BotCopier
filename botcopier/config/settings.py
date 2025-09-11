from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic_settings import BaseSettings


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

    model_config = {"env_prefix": "DATA_"}


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

    model_config = {"env_prefix": "TRAIN_"}


def save_params(
    data: DataConfig,
    training: TrainingConfig,
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
    path.write_text(yaml.safe_dump(existing, sort_keys=False))
