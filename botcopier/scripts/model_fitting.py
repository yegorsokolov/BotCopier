import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from pydantic import ValidationError  # type: ignore

try:  # pragma: no cover - optional dependency
    from schemas.trades import TradeEvent  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TradeEvent = None  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from botcopier.models.registry import get_model, register_model

# Optional model libraries and hyperparameter search
try:  # pragma: no cover - optional dependency
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import ray

    _HAS_RAY = True
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    _HAS_RAY = False

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import catboost as cb
except Exception:  # pragma: no cover - optional dependency
    cb = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.flight as flight
except Exception:  # pragma: no cover - optional dependency
    pa = None  # type: ignore
    flight = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore

if torch is not None:  # pragma: no cover - simple wrapper

    class FocalLoss(torch.nn.Module):
        """Binary Focal Loss for imbalanced classification."""

        def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
            super().__init__()
            self.gamma = gamma
            self.reduction = reduction

        def forward(
            self, input: "torch.Tensor", target: "torch.Tensor"
        ) -> "torch.Tensor":
            bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
            pt = torch.exp(-bce)
            loss = (1 - pt) ** self.gamma * bce
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss

else:  # pragma: no cover - torch optional

    class FocalLoss:  # type: ignore[misc]
        def __init__(self, *_, **__) -> None:
            raise ImportError("PyTorch is required for FocalLoss")


SCHEMA_VERSION = 3
START_EVENT_ID = 0


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight=None,
    C: float = 1.0,
    class_weight=None,
    existing_model: dict | None = None,
) -> LogisticRegression:
    """Fit a logistic regression model and return the classifier."""
    clf = LogisticRegression(
        max_iter=200,
        C=C,
        warm_start=existing_model is not None,
        class_weight=class_weight,
    )
    if existing_model is not None:
        clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
        clf.coef_ = np.array([existing_model.get("coefficients", [])])
        clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def fit_xgb_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight=None,
    existing_model: "xgb.XGBClassifier | None" = None,
    eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    early_stopping_rounds: int | None = None,
    **params: float | int,
) -> "xgb.XGBClassifier":  # pragma: no cover - optional dependency
    """Fit an ``xgboost.XGBClassifier`` and return the fitted model."""
    if xgb is None:  # pragma: no cover - optional dependency
        raise RuntimeError("xgboost is not available")
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
    }
    default_params.update(params)
    if early_stopping_rounds is not None:
        default_params["early_stopping_rounds"] = early_stopping_rounds
    clf = xgb.XGBClassifier(**default_params)
    if existing_model is not None:
        clf.load_model(existing_model)  # type: ignore[arg-type]
    fit_params: dict = {"sample_weight": sample_weight}
    if eval_set is not None:
        fit_params.update({"eval_set": eval_set, "verbose": False})
    clf.fit(X, y, **fit_params)
    best_iter = getattr(clf, "best_iteration", None)
    if best_iter is not None:
        logging.info("best_iteration=%s", best_iter)
    return clf


def fit_lgbm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight=None,
    existing_model: "lgb.LGBMClassifier | None" = None,
    eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    early_stopping_rounds: int | None = None,
    **params: float | int,
) -> "lgb.LGBMClassifier":  # pragma: no cover - optional dependency
    """Fit a ``lightgbm.LGBMClassifier`` and return the fitted model."""
    if lgb is None:  # pragma: no cover - optional dependency
        raise RuntimeError("lightgbm is not available")
    clf = lgb.LGBMClassifier(**params)
    if existing_model is not None:
        clf = existing_model
    fit_params: dict = {"sample_weight": sample_weight}
    if eval_set is not None and early_stopping_rounds is not None:
        fit_params.update(
            {
                "eval_set": eval_set,
                "early_stopping_rounds": early_stopping_rounds,
                "verbose": False,
            }
        )
    clf.fit(X, y, **fit_params)
    best_iter = getattr(clf, "best_iteration_", None)
    if best_iter is not None:
        logging.info("best_iteration=%s", best_iter)
    return clf


def fit_catboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight=None,
    existing_model: "cb.CatBoostClassifier | None" = None,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int | None = None,
    **params: float | int,
) -> "cb.CatBoostClassifier":  # pragma: no cover - optional dependency
    """Fit a ``catboost.CatBoostClassifier`` and return the fitted model."""
    if cb is None:  # pragma: no cover - optional dependency
        raise RuntimeError("catboost is not available")
    default_params = {"verbose": False}
    default_params.update(params)
    clf = cb.CatBoostClassifier(**default_params)
    if existing_model is not None:
        clf = existing_model
    fit_params: dict = {"sample_weight": sample_weight}
    if eval_set is not None and early_stopping_rounds is not None:
        fit_params.update(
            {
                "eval_set": eval_set,
                "early_stopping_rounds": early_stopping_rounds,
                "verbose": False,
            }
        )
    clf.fit(X, y, **fit_params)
    best_iter = getattr(clf, "best_iteration_", None)
    if best_iter is None and hasattr(clf, "get_best_iteration"):
        best_iter = clf.get_best_iteration()
    if best_iter is not None:
        logging.info("best_iteration=%s", best_iter)
    return clf


# Register model builders
register_model("logistic", fit_logistic_regression)
register_model("xgboost", fit_xgb_classifier)
register_model("lgbm", fit_lgbm_classifier)
register_model("catboost", fit_catboost_classifier)


def fit_quantile_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantiles: Iterable[float] = (0.05, 0.5, 0.95),
    model_type: str = "gbrt",
    **params: float | int,
) -> dict[float, object]:
    """Fit regressors for each quantile and return mapping of quantile to model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    quantiles : Iterable[float], optional
        Quantiles to fit, by default (0.05, 0.5, 0.95).
    model_type : str, optional
        ``"gbrt"`` to use :class:`GradientBoostingRegressor` or ``"linear"`` to use
        :class:`QuantileRegressor`.
    **params : float | int
        Additional keyword arguments passed to the regressor constructor.
    """

    models: dict[float, object] = {}
    for q in quantiles:
        if model_type == "linear":
            reg = QuantileRegressor(quantile=q, **params)
        else:
            reg = GradientBoostingRegressor(loss="quantile", alpha=q, **params)
        reg.fit(X, y)
        models[float(q)] = reg
    return models


def fit_heteroscedastic_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    model_type: str = "linear",
    **params: float | int,
) -> tuple[object, object]:
    """Fit a heteroscedastic regressor returning mean and log-variance models.

    This trains two separate models: one predicting the conditional mean of ``y``
    given ``X`` and another predicting the log of the conditional variance.  The
    variance model is trained on the squared residuals from the mean model.  The
    function supports simple linear regression as well as gradient boosting
    regressors.  More sophisticated models (e.g. neural networks) can be added
    by extending the ``model_type`` argument.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    sample_weight : np.ndarray | None, optional
        Optional sample weights.
    model_type : str, optional
        ``"gbrt"`` to use :class:`GradientBoostingRegressor` models or
        ``"linear"`` to use :class:`LinearRegression`.  The default is
        ``"linear"``.
    **params : float | int
        Additional keyword arguments passed to the regressor constructor.

    Returns
    -------
    tuple[object, object]
        ``(mean_model, log_var_model)`` where ``mean_model`` predicts the mean
        of ``y`` and ``log_var_model`` predicts the log variance.
    """

    if model_type == "gbrt":
        mean_model = GradientBoostingRegressor(**params)
        mean_model.fit(X, y, sample_weight=sample_weight)
        residuals = y - mean_model.predict(X)
        log_var_model = GradientBoostingRegressor(**params)
        log_var_model.fit(
            X,
            np.log(residuals**2 + 1e-6),
            sample_weight=sample_weight,
        )
    else:
        # Default to simple linear regression for compatibility with JSON
        # serialisation used elsewhere in the project.
        mean_model = LinearRegression().fit(X, y, sample_weight=sample_weight)
        residuals = y - mean_model.predict(X)
        log_var_model = LinearRegression().fit(
            X,
            np.log(residuals**2 + 1e-6),
            sample_weight=sample_weight,
        )

    return mean_model, log_var_model


def _compute_decay_weights(
    event_times: np.ndarray, half_life_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return age in days and exponential decay weights.

    Parameters
    ----------
    event_times : np.ndarray
        Array of event timestamps as ``datetime64``.
    half_life_days : float
        Half-life in days controlling the decay rate.
    """

    ref_time = event_times.max()
    age_days = (ref_time - event_times).astype("timedelta64[s]").astype(float) / (
        24 * 3600
    )
    weights = 0.5 ** (age_days / half_life_days)
    return age_days, weights


def load_logs_db(db_file: Path) -> pd.DataFrame:
    """Load log rows from a SQLite database."""

    conn = sqlite3.connect(db_file)
    try:
        query = "SELECT * FROM logs"
        params: tuple = ()
        if START_EVENT_ID > 0:
            query += " WHERE CAST(event_id AS INTEGER) > ?"
            params = (START_EVENT_ID,)
        df_logs = pd.read_sql_query(
            query, conn, params=params, parse_dates=["event_time", "open_time"]
        )
    finally:
        conn.close()

    df_logs.columns = [c.lower() for c in df_logs.columns]

    if "schema_version" in df_logs.columns:
        df_logs["schema_version"] = (
            pd.to_numeric(df_logs["schema_version"], errors="coerce")
            .fillna(SCHEMA_VERSION)
            .astype(int)
        )
        mismatch_mask = df_logs["schema_version"] != SCHEMA_VERSION
        if mismatch_mask.any():
            logging.warning(
                "Dropping %s rows with schema version != %s",
                mismatch_mask.sum(),
                SCHEMA_VERSION,
            )
            df_logs = df_logs[~mismatch_mask]
    else:
        logging.warning("schema_version column missing; assuming %s", SCHEMA_VERSION)
        df_logs["schema_version"] = SCHEMA_VERSION

    if "open_time" in df_logs.columns:
        df_logs["trade_duration"] = (
            (
                pd.to_datetime(df_logs["event_time"])
                - pd.to_datetime(df_logs["open_time"])
            )
            .dt.total_seconds()
            .fillna(0)
        )
    if "duration_sec" in df_logs.columns:
        df_logs["duration_sec"] = (
            pd.to_numeric(df_logs["duration_sec"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df_logs["duration_sec"] = df_logs.get("trade_duration", 0).astype(int)
    if "exit_reason" in df_logs.columns:
        df_logs["exit_reason"] = (
            df_logs["exit_reason"].fillna("").astype(str).str.upper()
        )
    else:
        df_logs["exit_reason"] = ""
    for col in [
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "commission",
        "swap",
        "trend_estimate",
        "trend_variance",
    ]:
        if col not in df_logs.columns:
            df_logs[col] = 0.0
        else:
            df_logs[col] = pd.to_numeric(df_logs[col], errors="coerce").fillna(0.0)
    if "is_anomaly" not in df_logs.columns:
        df_logs["is_anomaly"] = 0
    else:
        df_logs["is_anomaly"] = (
            pd.to_numeric(df_logs["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    if "action" in df_logs.columns:
        df_logs["action"] = df_logs["action"].fillna("").str.upper()
        df_logs = df_logs[
            (df_logs["action"] == "") | df_logs["action"].isin(valid_actions)
        ]

    invalid_rows = pd.DataFrame(columns=df_logs.columns)
    if "event_id" in df_logs.columns:
        dup_mask = df_logs.duplicated(subset="event_id", keep="first")
        if dup_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[dup_mask]])
            logging.warning("Dropping %s duplicate event_id rows", dup_mask.sum())
        df_logs = df_logs[~dup_mask]

    if set(["ticket", "action"]).issubset(df_logs.columns):
        crit_mask = (
            df_logs["ticket"].isna()
            | (df_logs["ticket"].astype(str) == "")
            | df_logs["action"].isna()
            | (df_logs["action"].astype(str) == "")
        )
        if crit_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[crit_mask]])
            logging.warning(
                "Dropping %s rows with missing ticket/action", crit_mask.sum()
            )
        df_logs = df_logs[~crit_mask]

    if "lots" in df_logs.columns:
        df_logs["lots"] = pd.to_numeric(df_logs["lots"], errors="coerce")
    if "price" in df_logs.columns:
        df_logs["price"] = pd.to_numeric(df_logs["price"], errors="coerce")
    unreal_mask = pd.Series(False, index=df_logs.index)
    if "lots" in df_logs.columns:
        unreal_mask |= df_logs["lots"] < 0
    if "price" in df_logs.columns:
        unreal_mask |= df_logs["price"].isna()
    if unreal_mask.any():
        invalid_rows = pd.concat([invalid_rows, df_logs[unreal_mask]])
        logging.warning(
            "Dropping %s rows with negative lots or NaN price", unreal_mask.sum()
        )
    df_logs = df_logs[~unreal_mask]

    if not invalid_rows.empty:
        invalid_file = db_file.with_name("invalid_rows.csv")
        try:
            invalid_rows.to_csv(invalid_file, index=False)
        except Exception:  # pragma: no cover - disk issues
            pass

    return df_logs


def load_logs(
    data_dir: Path,
    *,
    lite_mode: bool = False,
    chunk_size: int | None = None,
    flight_uri: str | None = None,
    kafka_brokers: str | None = None,
) -> tuple[pd.DataFrame | Iterable[pd.DataFrame], List[str], List[str]]:
    """Load log rows from ``data_dir``.

    ``MODIFY`` entries are retained alongside ``OPEN`` and ``CLOSE``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files or a SQLite ``.db`` file.
    """

    if kafka_brokers:
        raise NotImplementedError("kafka_brokers not supported in load_logs")

    if flight_uri:
        client = flight.FlightClient(flight_uri)
        desc = flight.FlightDescriptor.for_path("trades")
        info = client.get_flight_info(desc)
        reader = client.do_get(info.endpoints[0].ticket)
        if lite_mode or chunk_size:

            def _iter_batches():
                for batch in reader:
                    yield pa.Table.from_batches([batch]).to_pandas()

            return _iter_batches(), [], []
        table = reader.read_all()
        df = table.to_pandas()
        df.columns = [c.lower() for c in df.columns]
        return df, [], []

    if data_dir.suffix == ".db" or data_dir.suffix in {".sqlite", ".sqlite3"}:
        df = load_logs_db(data_dir)
        return df, [], []

    if data_dir.is_file():
        df = pd.read_csv(data_dir, sep=";", header=0, dtype=str, engine="python")
        df.columns = [c.lower() for c in df.columns]
        return df, [], []

    fields = [
        "schema_version",
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "sl_dist",
        "tp_dist",
        "sl_hit_dist",
        "tp_hit_dist",
        "executed_model_idx",
        "decision_id",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
        "open_time",
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "equity",
        "margin_level",
        "commission",
        "swap",
        "is_anomaly",
        "trend_estimate",
        "trend_variance",
        "exit_reason",
        "duration_sec",
    ]

    data_commits: List[str] = []
    data_checksums: List[str] = []

    metrics_file = data_dir / "metrics.csv"
    df_metrics = None
    key_col: str | None = None
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file, sep=";")
        df_metrics.columns = [c.lower() for c in df_metrics.columns]
        if "schema_version" in df_metrics.columns:
            df_metrics["schema_version"] = (
                pd.to_numeric(df_metrics["schema_version"], errors="coerce")
                .fillna(SCHEMA_VERSION)
                .astype(int)
            )
            m_mask = df_metrics["schema_version"] != SCHEMA_VERSION
            if m_mask.any():
                logging.warning(
                    "Dropping %s metric rows with schema version != %s",
                    m_mask.sum(),
                    SCHEMA_VERSION,
                )
                df_metrics = df_metrics[~m_mask]
        else:
            logging.warning(
                "schema_version column missing in metrics; assuming %s", SCHEMA_VERSION
            )
            df_metrics["schema_version"] = SCHEMA_VERSION
        if "magic" in df_metrics.columns:
            key_col = "magic"
        elif "model_id" in df_metrics.columns:
            key_col = "model_id"

    invalid_file = data_dir / "invalid_rows.csv"

    cs = chunk_size or (50000 if lite_mode else None)

    def iter_chunks():
        seen_ids: set[str] = set()
        invalid_rows: List[pd.DataFrame] = []
        for log_file in sorted(data_dir.glob("trades_*.csv")):
            reader = pd.read_csv(
                log_file,
                sep=";",
                header=0,
                dtype=str,
                chunksize=cs,
                engine="python",
            )
            if isinstance(reader, pd.DataFrame):
                reader = [reader]
            manifest_file = log_file.with_suffix(".manifest.json")
            if manifest_file.exists():
                try:
                    with open(manifest_file) as mf:
                        meta = json.load(mf)
                    commit = meta.get("commit")
                    checksum = meta.get("checksum")
                    if commit:
                        data_commits.append(str(commit))
                    if checksum:
                        data_checksums.append(str(checksum))
                except Exception:
                    pass
            for chunk in reader:
                chunk = chunk.reindex(columns=fields)
                chunk.columns = [c.lower() for c in chunk.columns]
                invalid = pd.DataFrame(columns=chunk.columns)
                chunk["schema_version"] = (
                    pd.to_numeric(chunk.get("schema_version"), errors="coerce")
                    .fillna(SCHEMA_VERSION)
                    .astype(int)
                )
                ver_mask = chunk["schema_version"] != SCHEMA_VERSION
                if ver_mask.any():
                    invalid = pd.concat([invalid, chunk[ver_mask]])
                    logging.warning(
                        "Found %s rows with schema version != %s; coercing",
                        ver_mask.sum(),
                        SCHEMA_VERSION,
                    )
                    chunk.loc[ver_mask, "schema_version"] = SCHEMA_VERSION
                chunk["event_time"] = pd.to_datetime(
                    chunk.get("event_time"), errors="coerce"
                )
                if "broker_time" in chunk.columns:
                    chunk["broker_time"] = pd.to_datetime(
                        chunk.get("broker_time"), errors="coerce"
                    )
                    chunk["broker_time"] = chunk["broker_time"].where(
                        ~chunk["broker_time"].isna(), None
                    )
                if "local_time" in chunk.columns:
                    chunk["local_time"] = pd.to_datetime(
                        chunk.get("local_time"), errors="coerce"
                    )
                    chunk["local_time"] = chunk["local_time"].where(
                        ~chunk["local_time"].isna(), None
                    )
                if "open_time" in chunk.columns:
                    chunk["open_time"] = pd.to_datetime(
                        chunk.get("open_time"), errors="coerce"
                    )
                    chunk["trade_duration"] = (
                        (chunk["event_time"] - chunk["open_time"])
                        .dt.total_seconds()
                        .fillna(0)
                    )
                else:
                    chunk["trade_duration"] = 0.0
                if "duration_sec" in chunk.columns:
                    chunk["duration_sec"] = (
                        pd.to_numeric(chunk.get("duration_sec"), errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                else:
                    chunk["duration_sec"] = chunk["trade_duration"].astype(int)
                for col in [
                    "book_bid_vol",
                    "book_ask_vol",
                    "book_imbalance",
                    "equity",
                    "margin_level",
                    "commission",
                    "swap",
                    "trend_estimate",
                    "trend_variance",
                ]:
                    chunk[col] = pd.to_numeric(
                        chunk.get(col, 0.0), errors="coerce"
                    ).fillna(0.0)
                chunk["exit_reason"] = (
                    chunk.get("exit_reason", "").fillna("").astype(str).str.upper()
                )
                chunk["is_anomaly"] = pd.to_numeric(
                    chunk.get("is_anomaly", 0), errors="coerce"
                ).fillna(0)
                for col in ["source", "comment"]:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].fillna("").astype(str)
                valid_actions = {"OPEN", "CLOSE", "MODIFY"}
                chunk["action"] = chunk["action"].fillna("").str.upper()
                chunk = chunk[
                    (chunk["action"] == "") | chunk["action"].isin(valid_actions)
                ]
                if "event_id" in chunk.columns:
                    dup_mask = chunk["event_id"].isin(seen_ids) | chunk.duplicated(
                        subset="event_id", keep="first"
                    )
                    if dup_mask.any():
                        invalid = pd.concat([invalid, chunk[dup_mask]])
                        logging.warning(
                            "Dropping %s duplicate event_id rows", dup_mask.sum()
                        )
                    seen_ids.update(chunk.loc[~dup_mask, "event_id"].tolist())
                    chunk = chunk[~dup_mask]
                if {"ticket", "action"}.issubset(chunk.columns):
                    crit_mask = (
                        chunk["ticket"].isna()
                        | (chunk["ticket"].astype(str) == "")
                        | chunk["action"].isna()
                        | (chunk["action"].astype(str) == "")
                    )
                    if crit_mask.any():
                        invalid = pd.concat([invalid, chunk[crit_mask]])
                        logging.warning(
                            "Dropping %s rows with missing ticket/action",
                            crit_mask.sum(),
                        )
                    chunk = chunk[~crit_mask]
                if "lots" in chunk.columns:
                    chunk["lots"] = pd.to_numeric(chunk["lots"], errors="coerce")
                if "price" in chunk.columns:
                    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                unreal_mask = pd.Series(False, index=chunk.index)
                if "lots" in chunk.columns:
                    unreal_mask |= chunk["lots"] < 0
                if "price" in chunk.columns:
                    unreal_mask |= chunk["price"].isna()
                if unreal_mask.any():
                    invalid = pd.concat([invalid, chunk[unreal_mask]])
                    logging.warning(
                        "Dropping %s rows with negative lots or NaN price",
                        unreal_mask.sum(),
                    )
                chunk = chunk[~unreal_mask]
                if "magic" in chunk.columns:
                    chunk["magic"] = (
                        pd.to_numeric(chunk["magic"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                if "executed_model_idx" in chunk.columns:
                    chunk["executed_model_idx"] = (
                        pd.to_numeric(chunk["executed_model_idx"], errors="coerce")
                        .fillna(-1)
                        .astype(int)
                    )
                if "decision_id" in chunk.columns:
                    chunk["decision_id"] = (
                        pd.to_numeric(chunk["decision_id"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                if (
                    df_metrics is not None
                    and key_col is not None
                    and "magic" in chunk.columns
                ):
                    if key_col == "magic":
                        chunk = chunk.merge(df_metrics, how="left", on="magic")
                    else:
                        chunk = chunk.merge(
                            df_metrics, how="left", left_on="magic", right_on="model_id"
                        )
                        chunk = chunk.drop(columns=["model_id"])
                if TradeEvent is not None:
                    records = []
                    for row in chunk.to_dict(orient="records"):
                        try:
                            TradeEvent(**row)
                            records.append(row)
                        except ValidationError:
                            invalid = pd.concat([invalid, pd.DataFrame([row])])
                    if records:
                        chunk = pd.DataFrame(records, columns=chunk.columns)
                    else:
                        chunk = pd.DataFrame(columns=chunk.columns)
                if not invalid.empty:
                    invalid_rows.append(invalid)
                yield chunk
        if invalid_rows:
            try:
                pd.concat(invalid_rows, ignore_index=True).to_csv(
                    invalid_file, index=False
                )
            except Exception:  # pragma: no cover - disk issues
                pass

    if lite_mode or chunk_size:
        return iter_chunks(), data_commits, data_checksums
    dfs = list(iter_chunks())
    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=[c.lower() for c in fields])
    return df_logs, data_commits, data_checksums


def compute_vif(X: np.ndarray, feature_names: Iterable[str]) -> pd.Series:
    """Return variance inflation factor for each feature.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    feature_names : Iterable[str]
        Names corresponding to the columns of ``X``.

    Returns
    -------
    pd.Series
        Series indexed by ``feature_names`` containing VIF values.
    """

    feature_names = list(feature_names)
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match number of columns in X")

    vifs: list[float] = []
    for i in range(X.shape[1]):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        if X_other.shape[1] == 0:
            vifs.append(1.0)
            continue
        reg = LinearRegression()
        reg.fit(X_other, y)
        r2 = reg.score(X_other, y)
        vif = float("inf") if np.isclose(1.0 - r2, 0.0) else 1.0 / (1.0 - r2)
        vifs.append(float(vif))

    return pd.Series(vifs, index=feature_names, name="VIF")


def scale_features(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Update ``scaler`` with ``X`` and return scaled data."""
    scaler.partial_fit(X)
    return scaler.transform(X)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    event_times: np.ndarray,
    out_dir: Path,
    *,
    half_life_days: float = 0.0,
    C: float = 1.0,
    model_types: Iterable[str] | None = None,
    n_trials: int = 20,
    distributed: bool = False,
    feature_names: Iterable[str] | None = None,
    vif_threshold: float = 10.0,
) -> dict:
    """Train a classifier with time-decayed sample weights.

    Data is split chronologically using ``TimeSeriesSplit`` so the validation
    set contains the most recent samples to avoid leakage.  When
    ``half_life_days`` is positive, sample weights are decayed exponentially
    based on the age of each trade.  When multiple ``model_types`` are provided
    and :mod:`optuna` is available, a hyperparameter search is conducted to pick
    the best performing model on the validation set.
    """

    if distributed and not _HAS_RAY:
        raise RuntimeError("ray is required for distributed execution")
    if model_types is None:
        model_types = ["logistic"]
    model_types = list(model_types)

    # Compute and filter features based on variance inflation factor
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    feature_names = list(feature_names)
    vifs = compute_vif(X, feature_names)
    mask = vifs <= vif_threshold
    if not mask.all():
        removed = vifs[~mask]
        logging.info(
            "Removing features with VIF > %s: %s",
            vif_threshold,
            ", ".join(f"{name}={val:.2f}" for name, val in removed.items()),
        )
        X = X[:, mask.to_numpy()]
        feature_names = [fn for fn, keep in zip(feature_names, mask.to_numpy()) if keep]

    # Filter out models whose libraries are unavailable
    available: dict[str, bool] = {
        "logistic": True,
        "xgboost": xgb is not None,
        "lgbm": lgb is not None,
        "catboost": cb is not None,
    }
    model_types = [m for m in model_types if available.get(m, False)]
    if not model_types:
        raise ValueError("No valid model types available")

    n_splits = max(1, min(5, len(y) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    sw = None
    if half_life_days > 0:
        _, decay = _compute_decay_weights(event_times[train_idx], half_life_days)
        sw = decay

    default_params: dict[str, dict[str, float | int]] = {
        "logistic": {"C": C},
        "xgboost": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "lgbm": {"learning_rate": 0.1, "max_depth": -1, "n_estimators": 100},
        "catboost": {"depth": 6, "learning_rate": 0.1, "iterations": 100},
    }

    suggest_funcs: dict[str, Callable[["optuna.Trial"], dict[str, float | int]]] = {
        "logistic": lambda t: {"C": t.suggest_float("C", 1e-3, 10.0, log=True)},
        "xgboost": lambda t: {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "n_estimators": t.suggest_int("n_estimators", 50, 200),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        },
        "lgbm": lambda t: {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", -1, 8),
            "n_estimators": t.suggest_int("n_estimators", 50, 200),
        },
        "catboost": lambda t: {
            "depth": t.suggest_int("depth", 2, 8),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": t.suggest_int("iterations", 50, 200),
        },
    }

    def _fit_single(model_type: str, trial: "optuna.Trial | None" = None):
        params = default_params.get(model_type, {}).copy()
        if trial is not None and model_type in suggest_funcs:
            params.update(suggest_funcs[model_type](trial))
        builder = get_model(model_type)
        clf = builder(X_train, y_train, sample_weight=sw, **params)
        preds = clf.predict(X_val)
        probas = (
            clf.predict_proba(X_val)[:, 1]
            if hasattr(clf, "predict_proba")
            else preds.astype(float)
        )
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        if len(np.unique(y_val)) > 1:
            roc = roc_auc_score(y_val, probas)
        else:  # pragma: no cover - single-class validation set
            roc = float("nan")
        metrics = {"accuracy": float(acc), "f1": float(f1), "roc_auc": float(roc)}
        return clf, params, metrics

    if optuna is not None and len(model_types) > 1:
        if distributed and _HAS_RAY:

            @ray.remote
            def objective(trial: "optuna.Trial") -> float:
                m_type = trial.suggest_categorical("model_type", model_types)
                clf, params, metrics = _fit_single(m_type, trial)
                trial.set_user_attr("model", clf)
                trial.set_user_attr("params", {**params, "model_type": m_type})
                trial.set_user_attr("metrics", metrics)
                return metrics["f1"]

            study = optuna.create_study(direction="maximize")

            def _objective(trial: "optuna.Trial") -> float:
                return ray.get(objective.remote(trial))

            study.optimize(_objective, n_trials=n_trials)
        else:

            def objective(trial: "optuna.Trial") -> float:
                m_type = trial.suggest_categorical("model_type", model_types)
                clf, params, metrics = _fit_single(m_type, trial)
                trial.set_user_attr("model", clf)
                trial.set_user_attr("params", {**params, "model_type": m_type})
                trial.set_user_attr("metrics", metrics)
                return metrics["f1"]

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

        trial = study.best_trial
        clf = trial.user_attrs["model"]
        params = trial.user_attrs["params"]
        metrics = trial.user_attrs["metrics"]
        best_type = params.pop("model_type")
        hyperparams = params
    else:
        # Fallback to first available model without search
        best_type = model_types[0]
        clf, params, metrics = _fit_single(best_type, None)
        hyperparams = params

    model: dict[str, object] = {
        "model_type": best_type,
        "hyperparameters": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in hyperparams.items()
        },
        "half_life_days": float(half_life_days),
        "validation_metrics": metrics,
        "feature_names": feature_names,
    }

    if best_type == "logistic":
        model.update(
            {
                "coefficients": clf.coef_[0].astype(float).tolist(),
                "intercept": float(clf.intercept_[0]),
                "classes": [int(c) for c in clf.classes_],
            }
        )

    if half_life_days > 0:
        model["weight_decay"] = {
            "half_life_days": float(half_life_days),
            "ref_time": np.datetime_as_string(event_times.max(), unit="s"),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f)

    return model


def main() -> None:
    """Command line interface for fitting a logistic regression model.

    The input dataset is expected to be an ``.npz`` file containing ``X``,
    ``y`` and ``event_times`` arrays.  The ``--half-life-days`` flag applies an
    exponential decay weight of ``0.5 ** (age_days / half_life_days)`` to each
    sample before fitting so that older trades influence the model less.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Fit classification model")
    parser.add_argument("dataset", help="NPZ file with X, y and event_times arrays")
    parser.add_argument("out_dir", help="Directory to write model.json")
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=0.0,
        help="half-life in days for sample weight decay",
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="inverse regularization strength"
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["logistic"],
        help="candidate models: logistic, xgboost, lgbm, catboost",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="number of Optuna trials when searching models",
    )
    args = parser.parse_args()

    data = np.load(args.dataset)
    out_dir = Path(args.out_dir)
    train_model(
        data["X"],
        data["y"],
        data["event_times"],
        out_dir,
        half_life_days=args.half_life_days,
        C=args.C,
        model_types=args.model_types,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
