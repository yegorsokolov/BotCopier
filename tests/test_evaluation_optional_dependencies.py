from pathlib import Path

import pytest

from botcopier.scripts import evaluation


def test_evaluate_requires_optional_dependencies(monkeypatch, tmp_path: Path) -> None:
    """``evaluate`` should raise when pandas/numpy are unavailable."""

    monkeypatch.setattr(evaluation, "pd", None)

    with pytest.raises(ImportError, match="optional dependencies not installed"):
        evaluation.evaluate(tmp_path / "pred.csv", tmp_path / "trades.csv", window=60)


def test_load_predictions_requires_numpy_for_variance(monkeypatch, tmp_path: Path) -> None:
    """Parsing variance without NumPy should raise the documented ImportError."""

    csv_path = tmp_path / "preds.csv"
    csv_path.write_text(
        "timestamp;symbol;direction;lots;probability;variance\n"
        "2024.01.01 00:00:00;EURUSD;buy;0.1;0.8;0.2\n"
    )

    monkeypatch.setattr(evaluation, "np", None)

    with pytest.raises(ImportError, match="optional dependencies not installed"):
        evaluation._load_predictions(csv_path)


def test_threshold_search_requires_numpy(monkeypatch) -> None:
    """Threshold search must surface the existing ImportError when NumPy is missing."""

    monkeypatch.setattr(evaluation, "np", None)

    with pytest.raises(ImportError, match="numpy is required for threshold search"):
        evaluation.search_decision_threshold([1.0], [0.5], [1.0])
