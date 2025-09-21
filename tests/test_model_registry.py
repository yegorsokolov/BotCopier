from __future__ import annotations

import errno
import json
import logging
from pathlib import Path

import pytest

from botcopier.models.registry import MODEL_VERSION, load_params
from botcopier.models.schema import ModelParams


@pytest.mark.usefixtures("caplog")
def test_load_params_skips_write_for_up_to_date_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = ModelParams(feature_names=["spread"])
    model_path = tmp_path / "model.json"
    model_path.write_text(params.model_dump_json())

    original_write_text = Path.write_text

    def fail_if_called(self: Path, *args, **kwargs):
        if self == model_path:
            raise PermissionError(errno.EACCES, "read-only path")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_if_called)

    caplog.set_level(logging.WARNING, logger="botcopier.models.registry")

    loaded = load_params(model_path)

    assert loaded.model_dump() == params.model_dump()
    assert not any(
        "Unable to update model parameters" in record.message for record in caplog.records
    )


@pytest.mark.usefixtures("caplog")
def test_load_params_warns_when_read_only_blocks_write(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_data = {"feature_names": ["spread"], "version": MODEL_VERSION - 1}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(legacy_data))

    original_write_text = Path.write_text
    write_calls: list[str] = []

    def fail_with_warning(self: Path, text: str, *args, **kwargs):
        if self == model_path:
            write_calls.append(text)
            raise PermissionError(errno.EACCES, "read-only path")
        return original_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_with_warning)

    caplog.set_level(logging.WARNING, logger="botcopier.models.registry")

    loaded = load_params(model_path)

    assert write_calls, "load_params should attempt to persist migrated parameters"
    assert loaded.version == MODEL_VERSION
    assert any(
        "Unable to update model parameters" in record.message for record in caplog.records
    )
