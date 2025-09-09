import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from scripts.model_fitting import (
    load_logs,
    scale_features,
    fit_xgb_classifier,
    fit_quantile_model,
)


def _write_log(path: Path) -> None:
    fields = [
        "schema_version",
        "event_id",
        "event_time",
        "action",
        "ticket",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
    ]
    rows = [
        ["1", "1", "2024.01.01 00:00:00", "OPEN", "1", "EURUSD", "0", "0.1", "1.1", "1.0", "1.2", "0", "0", "", "0.1", "0", "100"],
        ["1", "2", "2024.01.01 00:01:00", "OPEN", "2", "EURUSD", "0", "0.1", "1.1", "1.0", "1.2", "0", "0", "", "0.1", "0", "100"],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_load_logs_basic(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_log(log_dir / "trades_1.csv")
    df, commits, checksums = load_logs(log_dir)
    assert len(df) == 2
    assert commits == []
    assert checksums == []
    assert "event_time" in df.columns


def test_scale_features() -> None:
    scaler = StandardScaler()
    X = np.array([[0.0], [1.0], [2.0]])
    Xs = scale_features(scaler, X)
    assert Xs.shape == X.shape
    assert np.allclose(Xs.mean(), 0.0, atol=1e-7)


def test_early_stopping_reduces_overfit(caplog) -> None:
    pytest.importorskip("xgboost")
    import xgboost as xgb

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=0,
    )
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    clf_no = xgb.XGBClassifier(
        n_estimators=50,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    clf_no.fit(X_train, y_train)
    train_no = log_loss(y_train, clf_no.predict_proba(X_train)[:, 1])
    val_no = log_loss(y_val, clf_no.predict_proba(X_val)[:, 1])

    with caplog.at_level("INFO"):
        clf_es = fit_xgb_classifier(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
            n_estimators=50,
        )
    train_es = log_loss(y_train, clf_es.predict_proba(X_train)[:, 1])
    val_es = log_loss(y_val, clf_es.predict_proba(X_val)[:, 1])

    overfit_no = abs(train_no - val_no)
    overfit_es = abs(train_es - val_es)
    assert overfit_es < overfit_no
    assert any("best_iteration" in rec.message for rec in caplog.records)
    assert clf_es.best_iteration < 49


def test_fit_quantile_model_produces_monotonic_quantiles() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = X.ravel() + rng.normal(scale=0.1, size=100)
    models = fit_quantile_model(X, y, quantiles=(0.05, 0.5, 0.95))
    preds = {q: m.predict(X) for q, m in models.items()}
    assert set(models.keys()) == {0.05, 0.5, 0.95}
    medians = {q: float(np.median(p)) for q, p in preds.items()}
    assert medians[0.05] <= medians[0.5] <= medians[0.95]
