"""Tests for configuration loader and schema validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from jsonschema import ValidationError as JSONValidationError

from botcopier.config.settings import load_settings


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
