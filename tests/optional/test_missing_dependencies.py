import importlib
import logging
import sys
import types

import pytest


def test_missing_xgboost_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    sys.modules.pop("botcopier.models.registry", None)
    registry = importlib.import_module("botcopier.models.registry")
    with pytest.raises(ImportError, match="xgboost is required"):
        registry._fit_xgboost_classifier()
    monkeypatch.undo()
    importlib.reload(registry)


def test_missing_catboost_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "catboost", None)
    sys.modules.pop("botcopier.models.registry", None)
    registry = importlib.import_module("botcopier.models.registry")
    with pytest.raises(ImportError, match="catboost is required"):
        registry._fit_catboost_classifier()
    monkeypatch.undo()
    importlib.reload(registry)


def test_missing_torch_logs_info(monkeypatch, caplog):
    caplog.clear()
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "xgboost", types.ModuleType("xgboost"))
    monkeypatch.setitem(sys.modules, "catboost", types.ModuleType("catboost"))
    sys.modules.pop("botcopier.models.registry", None)
    with caplog.at_level(logging.INFO, logger="botcopier.models.registry"):
        registry = importlib.import_module("botcopier.models.registry")
    records = [
        record
        for record in caplog.records
        if record.name == "botcopier.models.registry"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    message = record.getMessage().lower()
    assert "pytorch" in message
    assert "disabled" in message
    assert "no module named" not in message
    monkeypatch.undo()
    importlib.reload(registry)


def test_missing_torch_logs_debug_details(monkeypatch, caplog):
    caplog.clear()
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "xgboost", types.ModuleType("xgboost"))
    monkeypatch.setitem(sys.modules, "catboost", types.ModuleType("catboost"))
    sys.modules.pop("botcopier.models.registry", None)
    with caplog.at_level(logging.DEBUG, logger="botcopier.models.registry"):
        registry = importlib.import_module("botcopier.models.registry")
    info_records = [
        record
        for record in caplog.records
        if record.name == "botcopier.models.registry" and record.levelno == logging.INFO
    ]
    debug_records = [
        record
        for record in caplog.records
        if record.name == "botcopier.models.registry" and record.levelno == logging.DEBUG
    ]
    assert info_records
    assert debug_records
    info_message = info_records[0].getMessage().lower()
    assert "pytorch" in info_message
    assert "disabled" in info_message
    debug_record = debug_records[0]
    debug_message = debug_record.getMessage().lower()
    assert "import error" in debug_message
    if debug_record.args:
        exc_text = str(debug_record.args[-1]).lower()
        assert exc_text in debug_message
    monkeypatch.undo()
    importlib.reload(registry)


def test_missing_polars_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", None)
    sys.modules.pop("botcopier.features.technical", None)
    technical = importlib.import_module("botcopier.features.technical")
    assert technical.pl is None
    assert not technical._HAS_POLARS
    monkeypatch.undo()
    importlib.reload(technical)
