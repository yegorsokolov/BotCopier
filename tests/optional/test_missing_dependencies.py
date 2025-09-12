import importlib
import sys

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


def test_missing_polars_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", None)
    sys.modules.pop("botcopier.features.technical", None)
    technical = importlib.import_module("botcopier.features.technical")
    assert technical.pl is None
    assert not technical._HAS_POLARS
    monkeypatch.undo()
    importlib.reload(technical)
