import json
from pathlib import Path

import pytest

from botcopier.training.pipeline import train


def test_cross_validation_metrics_written(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "1,1.0,1,1.0\n"
        "0,-0.5,2,1.1\n"
        "1,0.2,9,1.2\n"
        "0,-0.3,10,1.3\n"
        "1,0.4,17,1.4\n"
        "0,-0.6,18,1.5\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir, force_heavy=True)
    model = json.loads((out_dir / "model.json").read_text())
    assert "cv_metrics" in model
    assert "threshold" in model
    assert "cv_accuracy" in model and "cv_profit" in model
    assert "conformal_lower" in model and "conformal_upper" in model
    cv_metrics = model["cv_metrics"]
    assert cv_metrics["threshold_objective"] == "profit"
    assert cv_metrics["threshold"] == pytest.approx(model["threshold"])
    assert "brier_score" in cv_metrics
    assert "sharpe_ratio" in cv_metrics
    assert "ood_rate" in cv_metrics
    assert "ood_threshold" in cv_metrics
    for params in model["session_models"].values():
        assert params["cv_metrics"]
        assert "conformal_lower" in params and "conformal_upper" in params
        assert params["threshold"] == pytest.approx(model["threshold"])
        assert "metrics" in params
        assert params["metrics"]["threshold"] == pytest.approx(model["threshold"])
        assert "brier_score" in params["metrics"]
        assert "ood_rate" in params["metrics"]
        assert "ood_threshold" in params["metrics"]
        for fm in params["cv_metrics"]:
            assert "accuracy" in fm and "profit" in fm
            assert "ood_rate" in fm
            assert "ood_threshold" in fm


def test_training_fails_when_threshold_unmet(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "0,-1,1,1.0\n"
        "1,-2,2,1.0\n"
        "0,-1,9,1.0\n"
        "1,-2,10,1.0\n"
        "0,-1,17,1.0\n"
        "1,-2,18,1.0\n"
    )
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        train(
            data,
            out_dir,
            min_accuracy=1.1,
            min_profit=0.1,
            force_heavy=True,
        )


def test_ood_shrinkage_handles_rank_deficiency(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "1,1.0,3,1.0\n"
        "0,-0.5,3,1.1\n"
        "1,0.8,3,1.2\n"
        "0,-0.4,3,1.3\n"
        "1,0.6,3,1.4\n"
        "0,-0.3,3,1.5\n"
        "1,0.7,3,1.6\n"
        "0,-0.2,3,1.7\n"
    )
    out_dir = tmp_path / "out"
    train(
        data,
        out_dir,
        force_heavy=True,
        feature_subset=["spread", "hour_cos"],
    )
    model = json.loads((out_dir / "model.json").read_text())
    ood = model["ood"]
    cov = np.asarray(ood["covariance"], dtype=float)
    precision = np.asarray(ood.get("precision", []), dtype=float)
    assert cov.shape[0] == cov.shape[1] == 2
    assert precision.shape == cov.shape
    assert np.all(np.isfinite(cov))
    assert np.all(np.isfinite(precision))
    assert np.allclose(cov, cov.T, atol=1e-10)
    assert np.allclose(precision, precision.T, atol=1e-10)
    assert np.linalg.matrix_rank(cov) == cov.shape[0]
    sign, _ = np.linalg.slogdet(cov)
    assert sign > 0.0
    identity = np.eye(cov.shape[0])
    assert np.allclose(cov @ precision, identity, atol=1e-2)
    assert "ood_rate" in model["cv_metrics"]
    assert "ood_threshold" in model["cv_metrics"]
    assert model["cv_metrics"]["ood_rate"] >= 0.0
    for params in model["session_models"].values():
        assert "ood_rate" in params["metrics"]
        assert "ood_threshold" in params["metrics"]
        for fold_metrics in params["cv_metrics"]:
            assert "ood_rate" in fold_metrics
            assert "ood_threshold" in fold_metrics
