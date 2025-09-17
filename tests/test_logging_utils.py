"""Tests for the :mod:`logging_utils` helpers."""

from __future__ import annotations

import io
import json
import logging

import pytest

from logging_utils import setup_logging


@pytest.fixture(autouse=True)
def _reset_logging_state() -> None:
    """Ensure each test starts with a clean root logger configuration."""

    root = logging.getLogger()
    root.handlers.clear()
    if hasattr(root, "_botcopier_configured"):
        delattr(root, "_botcopier_configured")
    yield
    root.handlers.clear()
    if hasattr(root, "_botcopier_configured"):
        delattr(root, "_botcopier_configured")


def _stream_handler() -> logging.StreamHandler:
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            return handler
    raise AssertionError("stream handler was not configured")


def test_setup_logging_uses_json_formatter_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON formatting should be enabled when structured logs are requested."""

    stream = io.StringIO()
    monkeypatch.setenv("BOTCOPIER_JSON_LOGS", "1")

    logger = setup_logging("botcopier.tests")
    handler = _stream_handler()
    handler.setStream(stream)

    logger.info("structured log", extra={"foo": "bar"})
    handler.flush()

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "structured log"
    assert payload["foo"] == "bar"
    assert payload["logger"] == "botcopier.tests"
    assert payload["level"] == "INFO"


def test_setup_logging_supports_plain_formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``BOTCOPIER_LOG_FORMAT`` should enable text logging."""

    stream = io.StringIO()
    monkeypatch.delenv("BOTCOPIER_JSON_LOGS", raising=False)
    monkeypatch.setenv("BOTCOPIER_LOG_FORMAT", "plain")

    logger = setup_logging("botcopier.tests")
    handler = _stream_handler()
    handler.setStream(stream)

    logger.warning("plain text entry")
    handler.flush()

    log_line = stream.getvalue()
    assert "plain text entry" in log_line
    with pytest.raises(json.JSONDecodeError):
        json.loads(log_line)
