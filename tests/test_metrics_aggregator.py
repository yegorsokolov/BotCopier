"""Tests for the metrics aggregator persistence behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import metrics.aggregator as aggregator


def test_metrics_directory_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous_dir = aggregator.get_metrics_directory()
    monkeypatch.setenv("DATA_METRICS_DIR", str(tmp_path))
    try:
        configured = aggregator.configure_metrics_dir()
        assert configured == tmp_path.resolve()
        aggregator.add_metric("env", {"value": 1})
        payload = json.loads((tmp_path / "env.metrics.json").read_text())
        assert payload == [{"value": 1}]
    finally:
        aggregator.configure_metrics_dir(previous_dir)


def test_metrics_directory_read_only_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous_dir = aggregator.get_metrics_directory()
    read_only = tmp_path / "locked"
    read_only.mkdir()
    read_only.chmod(0o500)
    monkeypatch.setenv("DATA_METRICS_DIR", str(read_only))
    try:
        configured = aggregator.configure_metrics_dir()
        assert configured.resolve() != read_only.resolve()
        aggregator.add_metric("beta", 2)
        assert (configured / "beta.metrics.json").exists()
    finally:
        read_only.chmod(0o700)
        aggregator.configure_metrics_dir(previous_dir)


def test_persisted_metrics_reload(tmp_path: Path) -> None:
    previous_dir = aggregator.get_metrics_directory()
    metrics_dir = tmp_path / "persisted"
    metrics_dir.mkdir()
    initial = [{"value": 1}]
    (metrics_dir / "gamma.metrics.json").write_text(json.dumps(initial))
    try:
        aggregator.configure_metrics_dir(metrics_dir)
        aggregated = aggregator.get_aggregated_metrics()
        assert aggregated["gamma"] == initial
        aggregator.add_metric("gamma", {"value": 2})
        updated = json.loads((metrics_dir / "gamma.metrics.json").read_text())
        assert updated == [{"value": 1}, {"value": 2}]
    finally:
        aggregator.configure_metrics_dir(previous_dir)
