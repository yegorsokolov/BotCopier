import numpy as np
from pathlib import Path

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
