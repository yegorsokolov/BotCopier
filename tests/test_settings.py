"""Tests for configuration loader and schema validation."""

from pathlib import Path

import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError
from jsonschema import ValidationError as JSONValidationError

from botcopier.config.settings import (
    DataConfig,
    ExecutionConfig,
    TrainingConfig,
    compute_settings_hash,
    load_settings,
    save_params,
)


def test_load_settings_valid(tmp_path: Path) -> None:
    cfg = tmp_path / "params.yaml"
    cfg.write_text(
        """
data:
  csv: data.csv
training:
  batch_size: 16
execution:
  use_gpu: true
"""
    )
    data, training, execution = load_settings(path=cfg)
    assert data.csv == Path("data.csv")
    assert training.batch_size == 16
    assert execution.use_gpu is True


def test_load_settings_invalid_field(tmp_path: Path) -> None:
    cfg = tmp_path / "params.yaml"
    cfg.write_text(
        """
data:
  unknown: 1
"""
    )
    with pytest.raises((ValidationError, JSONValidationError)):
        load_settings(path=cfg)


def test_load_settings_missing_section(tmp_path: Path) -> None:
    cfg = tmp_path / "params.yaml"
    cfg.write_text(
        """
training:
  batch_size: 8
data: null
"""
    )
    with pytest.raises(TypeError):
        load_settings(path=cfg)


def test_save_params_persists_and_hash(tmp_path: Path) -> None:
    cfg_path = tmp_path / "params.yaml"
    data_cfg = DataConfig(csv=Path("trades.csv"))
    train_cfg = TrainingConfig(batch_size=64, model=Path("model.json"))
    exec_cfg = ExecutionConfig(use_gpu=True)

    digest = save_params(data_cfg, train_cfg, exec_cfg, path=cfg_path)
    data_loaded, train_loaded, exec_loaded = load_settings(path=cfg_path)
    assert data_loaded.csv == Path("trades.csv")
    assert train_loaded.batch_size == 64
    assert exec_loaded.use_gpu is True
    assert digest == compute_settings_hash(data_cfg, train_cfg, exec_cfg)
