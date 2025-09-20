import numpy as np
from pathlib import Path

from botcopier.models.registry import ConfidenceWeighted, get_model
from botcopier.scripts.online_trainer import OnlineTrainer


def _generate_stream():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(50, 1))
    y1 = (X1[:, 0] > 0).astype(int)
    X2 = rng.normal(size=(50, 1))
    y2 = (X2[:, 0] < 0).astype(int)  # concept drift: decision flips
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return X, y


def _train(trainer: OnlineTrainer, X: np.ndarray, y: np.ndarray) -> None:
    for xi, yi in zip(X, y):
        trainer.update([{"f0": float(xi[0]), "y": int(yi)}])


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    mean = float(arr.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        return np.ones_like(arr, dtype=float)
    return arr / mean


def test_confidence_weighted_updates_and_outperforms(tmp_path: Path) -> None:
    X, y = _generate_stream()
    # train confidence weighted model
    cw_path = tmp_path / "cw.json"
    trainer_cw = OnlineTrainer(cw_path, batch_size=1, run_generator=False, online_model="confidence_weighted")
    trainer_cw.clf.r = 0.1
    _train(trainer_cw, X, y)
    # ensure weights updated
    assert np.linalg.norm(trainer_cw.clf.w) > 0
    # save and reload to check variance persistence
    trainer_cw._save()
    w_before = trainer_cw.clf.w.copy()
    sigma_before = trainer_cw.clf.sigma.copy()
    trainer_cw2 = OnlineTrainer(cw_path, batch_size=1, run_generator=False, online_model="confidence_weighted")
    assert np.allclose(trainer_cw2.clf.w, w_before)
    assert np.allclose(trainer_cw2.clf.sigma, sigma_before)
    # accuracy after drift period
    X_last, y_last = X[-20:], y[-20:]
    acc_cw = (trainer_cw.clf.predict(X_last) == y_last).mean()

    # baseline SGD
    sgd_path = tmp_path / "sgd.json"
    trainer_sgd = OnlineTrainer(sgd_path, batch_size=1, run_generator=False)
    _train(trainer_sgd, X[:80], y[:80])
    acc_sgd = (trainer_sgd.clf.predict(X_last) == y_last).mean()

    assert acc_cw >= acc_sgd


def test_confidence_weighted_sample_weight_scales_updates() -> None:
    X = np.array([[0.2], [0.1], [-0.5], [-0.2]])
    y = np.array([1, 1, 0, 0])
    base = ConfidenceWeighted()
    base.partial_fit(X, y, classes=np.array([0, 1]))
    weighted = ConfidenceWeighted()
    weights = np.array([5.0, 1.0, 1.0, 1.0])
    weighted.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=weights)
    base_pos = base.predict_proba(X[[0]])[0, 1]
    weighted_pos = weighted.predict_proba(X[[0]])[0, 1]
    base_margin = base.confidence_score(X[[0]])[0]
    weighted_margin = weighted.confidence_score(X[[0]])[0]
    assert weighted_pos > base_pos
    assert weighted_margin > base_margin


def test_confidence_weighted_builder_normalises_weights() -> None:
    X = np.array([[0.2], [0.1], [-0.5], [-0.2]])
    y = np.array([1, 1, 0, 0])
    builder = get_model("confidence_weighted")
    meta_unweighted, predictor_unweighted = builder(X, y)
    weights = np.array([10.0, 1.0, 1.0, 1.0])
    meta_weighted, predictor_weighted = builder(X, y, sample_weight=weights)
    probs_unweighted = predictor_unweighted(X)
    probs_weighted = predictor_weighted(X)
    assert probs_weighted[0] > probs_unweighted[0]
    clf_weighted: ConfidenceWeighted = predictor_weighted.model  # type: ignore[assignment]
    expected_weights = _normalise_weights(weights)
    manual = ConfidenceWeighted()
    manual.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=expected_weights)
    assert np.allclose(clf_weighted.w, manual.w)
    assert np.allclose(clf_weighted.b, manual.b)
