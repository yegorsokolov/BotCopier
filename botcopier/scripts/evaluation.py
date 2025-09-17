from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence

try:  # pragma: no cover - optional parallel dependency
    from joblib import Parallel, delayed  # type: ignore

    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    _HAS_JOBLIB = False

# ``evaluate_strategy`` is intentionally lightweight and may be imported in
# environments where heavy scientific dependencies are absent.  The broader
# evaluation utilities rely on :mod:`numpy`, :mod:`pandas` and scikit-learn,
# so we attempt to import them but degrade gracefully when unavailable.
try:  # pragma: no cover - exercised in minimal test environments
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from sklearn.calibration import calibration_curve  # type: ignore
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    np = pd = None  # type: ignore
    calibration_curve = None  # type: ignore

    def _missing(*args, **kwargs):  # type: ignore
        raise ImportError("optional dependencies not installed")

    accuracy_score = (
        average_precision_score
    ) = brier_score_loss = roc_auc_score = _missing  # type: ignore

from botcopier.exceptions import DataError, ServiceError
from botcopier.scripts.stress_test import run_stress_tests, summarise_stress_results
from botcopier.utils.validation import validate_columns
from metrics.registry import get_metrics, load_plugins, register_metric
from schemas.decisions import DECISION_SCHEMA
from schemas.trades import TRADE_SCHEMA

logger = logging.getLogger(__name__)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Return an effective ``n_jobs`` respecting optional dependencies."""

    if not n_jobs or n_jobs <= 1 or not _HAS_JOBLIB:
        return 1
    return int(n_jobs)


def _vectorised_drawdown_matrix(returns: "np.ndarray") -> "np.ndarray":
    """Compute column-wise absolute drawdown for a returns matrix."""

    if np is None:  # pragma: no cover - optional dependency guard
        raise ImportError("numpy is required for vectorised drawdown computation")
    if returns.ndim == 1:
        returns = returns[:, None]
    cumulative = np.cumsum(returns, axis=0)
    cumulative = np.vstack([np.zeros((1, cumulative.shape[1])), cumulative])
    peaks = np.maximum.accumulate(cumulative, axis=0)
    drawdowns = peaks - cumulative
    return np.max(drawdowns, axis=0)


def _vectorised_var95_matrix(returns: "np.ndarray") -> "np.ndarray":
    """Compute column-wise 5%% value-at-risk using deterministic indexing."""

    if np is None:  # pragma: no cover - optional dependency guard
        raise ImportError("numpy is required for vectorised VaR computation")
    if returns.ndim == 1:
        returns = returns[:, None]
    ordered = np.sort(returns, axis=0)
    idx = int(0.05 * ordered.shape[0])
    idx = min(max(idx, 0), ordered.shape[0] - 1)
    return ordered[idx, :]


def _as_float_array(data: Sequence[float] | "np.ndarray") -> "np.ndarray":
    """Convert ``data`` to a 1-D ``float`` NumPy array."""

    if np is None:  # pragma: no cover - optional dependency guard
        raise ImportError("numpy is required for vectorised evaluation utilities")
    if isinstance(data, np.ndarray):
        return data.astype(float, copy=False)
    return np.asarray(list(data), dtype=float)


def _parse_time(value: str, *, symbol: str = "") -> datetime:
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise DataError("unrecognised time format", symbol=symbol, timestamp=value)


def _load_predictions(pred_file: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(pred_file, delimiter=";")
    except OSError:
        err = DataError(
            f"Failed to read predictions {pred_file}", timestamp=datetime.now(UTC)
        )
        logger.exception("%s; using empty DataFrame", err)
        cols = [
            "timestamp",
            "symbol",
            "direction",
            "lots",
            "probability",
            "value",
            "log_variance",
            "executed_model_idx",
            "decision_id",
        ]
        return pd.DataFrame(columns=cols)

    validate_columns(
        df,
        DECISION_SCHEMA,
        required={"timestamp", "probability"},
        name="predictions",
    )

    ts_col = next((c for c in ["timestamp", "time"] if c in df.columns), df.columns[0])
    sym_col = "symbol" if "symbol" in df.columns else None
    if sym_col is None:
        df["symbol"] = ""
        sym_col = "symbol"
    df["timestamp"] = df.apply(
        lambda r: _parse_time(str(r[ts_col]), symbol=str(r.get(sym_col, ""))), axis=1
    )

    dir_col = next(
        (c for c in ["direction", "order_type"] if c in df.columns), "direction"
    )
    df["direction"] = (
        df.get(dir_col, "")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "buy": 1, "0": -1, "-1": -1, "sell": -1})
        .fillna(1)
        .astype(int)
    )

    df["lots"] = pd.to_numeric(df.get("lots", 0), errors="coerce").fillna(0.0)

    prob_col = next(
        (c for c in ["probability", "prob", "proba"] if c in df.columns), None
    )
    df["probability"] = (
        pd.to_numeric(df[prob_col], errors="coerce") if prob_col else np.nan
    )

    val_col = next(
        (c for c in ["expected_value", "value", "pnl"] if c in df.columns), None
    )
    df["value"] = pd.to_numeric(df[val_col], errors="coerce") if val_col else np.nan

    if "log_variance" in df.columns or "logvar" in df.columns:
        log_col = "log_variance" if "log_variance" in df.columns else "logvar"
        df["log_variance"] = pd.to_numeric(df[log_col], errors="coerce")
    elif "variance" in df.columns:
        df["log_variance"] = np.log(pd.to_numeric(df["variance"], errors="coerce"))
    else:
        df["log_variance"] = np.nan

    model_col = next(
        (c for c in ["executed_model_idx", "model_idx", "model"] if c in df.columns),
        None,
    )
    if model_col:
        df["executed_model_idx"] = pd.to_numeric(df[model_col], errors="coerce")
    else:
        df["executed_model_idx"] = np.nan

    if "decision_id" in df.columns:
        df["decision_id"] = pd.to_numeric(df["decision_id"], errors="coerce")
    else:
        df["decision_id"] = np.nan

    cols = [
        "timestamp",
        "symbol",
        "direction",
        "lots",
        "probability",
        "value",
        "log_variance",
        "executed_model_idx",
        "decision_id",
    ]
    return df[cols]


def _load_actual_trades(log_file: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(log_file, delimiter=";")
    except OSError:
        err = DataError(
            f"Failed to read trade log {log_file}", timestamp=datetime.now(UTC)
        )
        logger.exception("%s; using empty DataFrame", err)
        cols = [
            "open_time",
            "close_time",
            "symbol",
            "direction",
            "lots",
            "executed_model_idx",
            "decision_id",
            "ticket",
            "profit",
        ]
        return pd.DataFrame(columns=cols)

    validate_columns(
        df,
        TRADE_SCHEMA,
        required={
            "event_time",
            "action",
            "ticket",
            "symbol",
            "order_type",
            "lots",
            "profit",
        },
        name="trade log",
    )

    ts_col = next(
        (c for c in ["event_time", "time_event"] if c in df.columns), df.columns[0]
    )
    sym_col = "symbol" if "symbol" in df.columns else None
    if sym_col is None:
        df["symbol"] = ""
        sym_col = "symbol"
    df["timestamp"] = df.apply(
        lambda r: _parse_time(str(r[ts_col]), symbol=str(r.get(sym_col, ""))), axis=1
    )
    df["action"] = df.get("action", "").astype(str).str.upper()
    df["ticket"] = df.get("ticket").astype(str)

    open_df = df[df["action"] == "OPEN"].copy()
    close_df = df[df["action"] == "CLOSE"].copy()

    open_df = open_df.rename(columns={"timestamp": "open_time"})
    if "profit" in open_df.columns:
        open_df = open_df.drop(columns=["profit"])
    open_df["direction"] = (
        pd.to_numeric(open_df.get("order_type", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        .map({0: 1})
        .fillna(-1)
        .astype(int)
    )
    open_df["lots"] = pd.to_numeric(open_df.get("lots", 0), errors="coerce").fillna(0.0)
    open_df["executed_model_idx"] = pd.to_numeric(
        open_df.get("executed_model_idx"), errors="coerce"
    )
    open_df["decision_id"] = pd.to_numeric(open_df.get("decision_id"), errors="coerce")

    close_df = close_df.rename(columns={"timestamp": "close_time"})
    close_df["profit"] = pd.to_numeric(
        close_df.get("profit", 0), errors="coerce"
    ).fillna(0.0)

    trades = pd.merge(
        open_df,
        close_df[["ticket", "close_time", "profit"]],
        on="ticket",
        how="inner",
    )
    cols = [
        "open_time",
        "close_time",
        "symbol",
        "direction",
        "lots",
        "executed_model_idx",
        "decision_id",
        "ticket",
        "profit",
    ]
    return trades[cols]


def evaluate(
    pred_file: Path,
    actual_log: Path,
    window: int,
    model_json: Optional[Path] = None,
    *,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    hooks: Sequence[str] | None = None,
) -> Dict:
    predictions = _load_predictions(pred_file)
    trades = _load_actual_trades(actual_log)

    conformal_lower = None
    conformal_upper = None
    model_value_mean = None
    model_value_std = None
    model_value_atoms = None
    model_value_dist = None
    model_value_quantiles = None
    if model_json is not None:
        try:
            with open(model_json) as f:
                m = json.load(f)
            conformal_lower = m.get("conformal_lower")
            conformal_upper = m.get("conformal_upper")
            model_value_mean = m.get("value_mean")
            model_value_std = m.get("value_std")
            model_value_atoms = m.get("value_atoms")
            model_value_dist = m.get("value_distribution")
            model_value_quantiles = m.get("value_quantiles")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read model metadata %s: %s", model_json, exc)

    # Initial merge on decision_id when available
    if predictions["decision_id"].notna().any() and trades["decision_id"].notna().any():
        merged = predictions.merge(
            trades[["decision_id", "open_time", "close_time", "profit", "ticket"]],
            on="decision_id",
            how="left",
        )
    else:
        merged = predictions.copy()
        merged["open_time"] = pd.NaT
        merged["close_time"] = pd.NaT
        merged["profit"] = np.nan
        merged["ticket"] = pd.NA
    matched_tickets = merged["ticket"].dropna().unique()

    # Match remaining predictions by timestamp within window
    unmatched_mask = merged["ticket"].isna()
    if unmatched_mask.any():
        remaining_trades = trades[~trades["ticket"].isin(matched_tickets)]
        pred_unmatched = merged.loc[unmatched_mask, predictions.columns]
        asof = pd.merge_asof(
            pred_unmatched.sort_values("timestamp"),
            remaining_trades.sort_values("open_time"),
            left_on="timestamp",
            right_on="open_time",
            by=["symbol", "direction"],
            tolerance=pd.Timedelta(seconds=window),
            direction="forward",
        )
        asof_nonan = asof.dropna(subset=["ticket"])
        asof_unique = asof_nonan.sort_values("timestamp").drop_duplicates(
            "ticket", keep="first"
        )
        merged.loc[asof_unique.index, "open_time"] = asof_unique["open_time"].values
        merged.loc[asof_unique.index, "close_time"] = asof_unique["close_time"].values
        merged.loc[asof_unique.index, "profit"] = asof_unique["profit"].values
        merged.loc[asof_unique.index, "ticket"] = asof_unique["ticket"].values
        matched_tickets = np.concatenate(
            [matched_tickets, asof_unique["ticket"].unique()]
        )

    unused_trades = trades[~trades["ticket"].isin(matched_tickets)]

    merged = merged.copy()
    merged["is_match"] = merged["ticket"].notna()
    matched_df = merged.loc[merged["is_match"]].copy()
    profits = matched_df["profit"].astype(float)
    net_profits = profits - (
        fee_per_trade + profits.abs() * slippage_bps * 1e-4
    )
    trade_times = matched_df["close_time"]

    tp = len(matched_df)
    matches = tp
    fp = len(predictions) - tp
    fn = len(trades) - tp
    total = tp + fp + fn
    accuracy = tp / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    gross_profit = float(profits[profits >= 0].sum())
    gross_loss = float(-profits[profits < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy = (gross_profit - gross_loss) / tp if tp else 0.0
    expected_return = float(profits.mean()) if tp else 0.0
    expected_return_net = float(net_profits.mean()) if tp else 0.0
    downside = profits[profits < 0]
    downside_net = net_profits[net_profits < 0]
    downside_risk = float(-downside.mean()) if not downside.empty else 0.0
    risk_reward = expected_return - downside_risk

    profit_array = profits.to_numpy(dtype=float)
    net_profit_array = net_profits.to_numpy(dtype=float)

    sharpe = sortino = 0.0
    if profit_array.size > 1:
        mean_profit = float(np.mean(profit_array))
        std_profit = float(np.std(profit_array, ddof=1))
        if std_profit > 0:
            sharpe = mean_profit / std_profit
        downside_arr = profit_array[profit_array < 0]
        if downside_arr.size:
            downside_std = float(np.sqrt(np.mean(downside_arr**2)))
            if downside_std > 0:
                sortino = mean_profit / downside_std

    sharpe_net = sortino_net = 0.0
    if net_profit_array.size > 1:
        mean_net = float(np.mean(net_profit_array))
        std_net = float(np.std(net_profit_array, ddof=1))
        if std_net > 0:
            sharpe_net = mean_net / std_net
        downside_net_arr = net_profit_array[net_profit_array < 0]
        if downside_net_arr.size:
            downside_net_std = float(np.sqrt(np.mean(downside_net_arr**2)))
            if downside_net_std > 0:
                sortino_net = mean_net / downside_net_std

    max_dd = _abs_drawdown(profit_array) if profit_array.size else 0.0
    max_dd_net = _abs_drawdown(net_profit_array) if net_profit_array.size else 0.0

    predictions_per_model = (
        merged["executed_model_idx"].dropna().astype(int).value_counts().to_dict()
    )
    matches_per_model = (
        matched_df["executed_model_idx"].dropna().astype(int).value_counts().to_dict()
    )

    mask_probs = merged["probability"].notna()
    prob_scores = merged.loc[mask_probs, "probability"].astype(float).fillna(0.0)
    y_true_probs = merged.loc[mask_probs, "is_match"].astype(int)
    extra_true = np.ones(len(unused_trades), dtype=int)
    extra_scores = np.zeros(len(unused_trades), dtype=float)
    y_true_vector = np.concatenate([y_true_probs.to_numpy(dtype=int), extra_true])
    y_score_vector = np.concatenate([prob_scores.to_numpy(dtype=float), extra_scores])

    bound_in = bound_total = 0
    if conformal_lower is not None and conformal_upper is not None:
        within = prob_scores.between(conformal_lower, conformal_upper)
        bound_total = int(mask_probs.sum())
        bound_in = int(within.sum())

    mu = matched_df["value"]
    log_v = matched_df["log_variance"]
    valid = mu.notna() & log_v.notna()
    if valid.any():
        mu_vals = mu[valid].to_numpy(dtype=float)
        log_v_vals = log_v[valid].to_numpy(dtype=float)
        profits_valid = profits[valid].to_numpy(dtype=float)
        var = np.exp(log_v_vals)
        nll_values = 0.5 * (
            np.log(2 * np.pi) + log_v_vals + ((profits_valid - mu_vals) ** 2) / var
        )
        nd = NormalDist()
        z = nd.inv_cdf(0.05)
        c = nd.pdf(z) / 0.05
        es_pred_values = mu_vals - np.sqrt(var) * c
    else:
        nll_values = np.array([], dtype=float)
        es_pred_values = np.array([], dtype=float)

    if tp:
        sorted_profits = np.sort(profit_array)
        n_tail = max(1, int(np.ceil(sorted_profits.size * 0.05)))
        tail = sorted_profits[:n_tail]
        cvar = float(tail.mean())
        var_95 = float(np.quantile(profit_array, 0.05))
        es_95 = float(profit_array[profit_array <= var_95].mean())
        var_95_net = float(np.quantile(net_profit_array, 0.05))
        net_tail = net_profit_array[net_profit_array <= var_95_net]
        es_95_net = float(net_tail.mean()) if net_tail.size else float(var_95_net)
    else:
        cvar = var_95 = es_95 = 0.0
        var_95_net = es_95_net = 0.0

    roc_auc = pr_auc = brier = ece = None
    reliability = {"prob_true": [], "prob_pred": []}
    if y_score_vector.size:
        brier = brier_score_loss(y_true_vector, y_score_vector)
        try:
            prob_true, prob_pred = calibration_curve(
                y_true_vector, y_score_vector, n_bins=10
            )
            reliability = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            }
            bins = np.linspace(0.0, 1.0, 11)
            binids = np.digitize(y_score_vector, bins[1:-1], right=True)
            bin_counts = np.bincount(binids, minlength=10)
            counts = bin_counts[bin_counts > 0]
            ece = float(
                np.sum(np.abs(prob_true - prob_pred) * (counts / len(y_score_vector)))
            )
        except ValueError:
            pass
        if len(np.unique(y_true_vector)) > 1:
            roc_auc = roc_auc_score(y_true_vector, y_score_vector)
            pr_auc = average_precision_score(y_true_vector, y_score_vector)

    conformal = bound_in / bound_total if bound_total else None
    nll_mean = float(nll_values.mean()) if nll_values.size else None
    es_pred_mean = float(es_pred_values.mean()) if es_pred_values.size else None

    stats: Dict[str, object] = {
        "matched_events": matches,
        "predicted_events": len(predictions),
        "actual_events": len(trades),
        "accuracy": accuracy,
        "recall": recall,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy": expectancy,
        "expected_return": expected_return,
        "expected_return_net": expected_return_net,
        "downside_risk": downside_risk,
        "max_drawdown": float(max_dd),
        "max_drawdown_net": float(max_dd_net),
        "cvar": cvar,
        "var_95": var_95,
        "es_95": es_95,
        "var_95_net": var_95_net,
        "es_95_net": es_95_net,
        "risk_reward": risk_reward,
        "conformal_coverage": conformal,
        "predictions_per_model": predictions_per_model,
        "matches_per_model": matches_per_model,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "sharpe_ratio_net": sharpe_net,
        "sortino_ratio_net": sortino_net,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "reliability_curve": reliability,
    }
    if nll_mean is not None:
        stats["neg_log_likelihood"] = nll_mean
    if es_pred_mean is not None:
        stats["expected_shortfall_pred"] = es_pred_mean
    if model_value_mean is not None:
        stats["model_value_mean"] = model_value_mean
    if model_value_std is not None:
        stats["model_value_std"] = model_value_std
    if model_value_atoms and model_value_dist:
        stats["model_value_atoms"] = model_value_atoms
        stats["model_value_distribution"] = model_value_dist
    if model_value_quantiles:
        stats["model_value_quantiles"] = model_value_quantiles

    order_types = (
        matched_df["order_type"].astype(str).tolist()
        if "order_type" in matched_df
        else None
    )
    stress_results = run_stress_tests(profits.tolist(), order_types=order_types)
    stats["stress_tests"] = {
        name: asdict(result) for name, result in stress_results.items()
    }
    stats["stress_summary"] = summarise_stress_results(stress_results)
    ctx = {
        "tp": tp,
        "fp": fp,
        "matches": matches,
        "profits": profits,
        "net_profits": net_profits,
        "downside": downside,
        "downside_net": downside_net,
        "trade_times": trade_times,
        "stats": stats,
    }
    from botcopier.eval.hooks import dispatch_hooks, load_entry_point_hooks

    load_entry_point_hooks(hooks)
    dispatch_hooks(hooks, ctx)

    return ctx["stats"]


@register_metric("accuracy")
def _metric_accuracy(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    preds = (probas >= threshold).astype(int)
    return float(accuracy_score(y_true, preds))


@register_metric("roc_auc")
def _metric_roc_auc(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float | None:
    return (
        float(roc_auc_score(y_true, probas)) if len(set(y_true.tolist())) > 1 else None
    )


@register_metric("pr_auc")
def _metric_pr_auc(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float | None:
    return (
        float(average_precision_score(y_true, probas))
        if len(set(y_true.tolist())) > 1
        else None
    )


@register_metric("brier_score")
def _metric_brier(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    return float(brier_score_loss(y_true, probas))


@register_metric("reliability_curve")
def _metric_reliability(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> Dict[str, List[float]]:
    try:
        prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10)
        return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    except ValueError:
        return {"prob_true": [], "prob_pred": []}


@register_metric("ece")
def _metric_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    """Expected calibration error."""
    bins = np.linspace(0.0, 1.0, 11)
    binids = np.digitize(probas, bins[1:-1], right=True)
    bin_counts = np.bincount(binids, minlength=10)
    try:
        prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10)
        counts = bin_counts[bin_counts > 0]
        ece = np.sum(np.abs(prob_true - prob_pred) * (counts / probas.shape[0]))
        return float(ece)
    except ValueError:
        return 0.0


def _returns(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    preds = (probas >= threshold).astype(int)
    if profits is None:
        return np.where(preds == y_true, 1.0, -1.0)
    return profits


@register_metric("sharpe_ratio")
def _metric_sharpe(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    returns = _returns(y_true, probas, profits, threshold=threshold)
    if returns.size > 1:
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        if std > 0:
            return mean / std
    return 0.0


@register_metric("sortino_ratio")
def _metric_sortino(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    returns = _returns(y_true, probas, profits, threshold=threshold)
    if returns.size > 0:
        mean = float(np.mean(returns))
        downside = returns[returns < 0]
        if downside.size > 0:
            downside_std = float(np.sqrt(np.mean(downside**2)))
            if downside_std > 0:
                return mean / downside_std
    return 0.0


@register_metric("max_drawdown")
def _metric_drawdown(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    returns = _returns(y_true, probas, profits, threshold=threshold).tolist()
    return float(_abs_drawdown(returns))


@register_metric("var_95")
def _metric_var95(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    returns = _returns(y_true, probas, profits, threshold=threshold).tolist()
    return float(_var_95(returns))


@register_metric("profit")
def _metric_profit(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    returns = _returns(y_true, probas, profits, threshold=threshold)
    return float(np.sum(returns))


def _classification_metrics(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    selected: Sequence[str] | None = None,
    *,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute classification and calibration metrics."""

    load_plugins(selected)
    results: Dict[str, object] = {}
    for name, fn in get_metrics(selected).items():
        try:
            results[name] = fn(y_true, probas, profits, threshold=threshold)
        except TypeError:
            results[name] = fn(y_true, probas, profits)
        except (
            Exception
        ) as exc:  # pragma: no cover - specific errors tested via injection
            err = ServiceError(f"Metric {name} failed", timestamp=datetime.now(UTC))
            logger.warning("%s", err, exc_info=exc)
            results[name] = np.nan
    return results


def _candidate_thresholds(
    probas: np.ndarray, threshold_grid: Sequence[float] | None
) -> np.ndarray:
    """Return sorted unique decision threshold candidates."""

    if threshold_grid is None:
        grid = np.asarray([], dtype=float)
    else:
        try:
            grid = np.asarray(list(threshold_grid), dtype=float)
        except TypeError:
            grid = np.asarray([float(threshold_grid)], dtype=float)
    finite_probs = probas[np.isfinite(probas)]
    candidates = np.concatenate(
        [grid, finite_probs, np.asarray([0.0, 0.5, 1.0], dtype=float)]
    )
    candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
    if candidates.size == 0:
        return np.asarray([0.5], dtype=float)
    uniq = np.unique(candidates)
    if uniq.size > 512:
        idx = np.linspace(0, uniq.size - 1, 512, dtype=int)
        uniq = uniq[idx]
    return uniq


def search_decision_threshold(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None,
    *,
    objective: str = "profit",
    threshold_grid: Sequence[float] | None = None,
    metric_names: Sequence[str] | None = None,
    max_drawdown: float | None = None,
    var_limit: float | None = None,
    n_jobs: int | None = None,
) -> tuple[float, Dict[str, object]]:
    """Search for a decision threshold that maximises ``objective``.

    Parameters
    ----------
    y_true:
        Ground truth binary labels.
    probas:
        Predicted probabilities for the positive class.
    profits:
        Per-sample returns used for financial metrics.  If ``None`` the
        threshold search operates on a synthetic +1/-1 payoff based on
        classification correctness.
    objective:
        Optimisation objective.  Supported values: ``profit`` (default),
        ``net_profit`` (alias of ``profit``), ``sharpe`` and ``sortino``.
    threshold_grid:
        Optional iterable of thresholds to evaluate in addition to the values
        present in ``probas``.
    metric_names:
        Optional sequence restricting the computed metrics.
    max_drawdown, var_limit:
        Risk limits.  Thresholds producing metrics that breach these limits are
        discarded.  If all thresholds breach the limits a ``ValueError`` is
        raised.

    Returns
    -------
    tuple
        ``(best_threshold, metrics)`` for the selected threshold.
    """

    if np is None:  # pragma: no cover - optional dependency guard
        raise ImportError("numpy is required for threshold search")

    y_true = np.asarray(y_true, dtype=float)
    probas = np.asarray(probas, dtype=float)
    if profits is None:
        profits = np.where(y_true == 1.0, 1.0, -1.0)
    profits = np.asarray(profits, dtype=float)

    candidates = _candidate_thresholds(probas, threshold_grid)
    thresholds = candidates.astype(float)
    objective_metric_map = {
        "profit": "profit",
        "net_profit": "profit",
        "sharpe": "sharpe_ratio",
        "sortino": "sortino_ratio",
    }
    metric_key = objective_metric_map.get(objective, objective)
    returns_matrix = profits[:, None] * (probas[:, None] >= thresholds[None, :])
    drawdowns = _vectorised_drawdown_matrix(returns_matrix)
    var95_values = _vectorised_var95_matrix(returns_matrix)

    def _compute(idx: int) -> tuple[float, Dict[str, object]]:
        thr = float(thresholds[idx])
        returns = returns_matrix[:, idx]
        metrics = _classification_metrics(
            y_true,
            probas,
            returns,
            selected=metric_names,
            threshold=thr,
        )
        metrics.setdefault("max_drawdown", float(drawdowns[idx]))
        metrics.setdefault("var_95", float(var95_values[idx]))
        metrics["threshold"] = thr
        metrics["threshold_objective"] = objective
        return thr, metrics

    effective_jobs = _resolve_n_jobs(n_jobs)
    indices = range(len(thresholds))
    if len(thresholds) == 1 or effective_jobs == 1 or not _HAS_JOBLIB:
        evaluated = [_compute(i) for i in indices]
    else:
        evaluated = Parallel(n_jobs=effective_jobs, prefer="threads")(
            delayed(_compute)(i) for i in indices
        )

    thresholds_eval = np.asarray([thr for thr, _ in evaluated], dtype=float)
    metrics_list = [metrics for _, metrics in evaluated]

    def _metric_value(metrics: Dict[str, object]) -> float:
        val = metrics.get(metric_key)
        if isinstance(val, (int, float)):
            val = float(val)
            if not math.isnan(val):
                return val
        return float("nan")

    metric_values = np.fromiter(
        (_metric_value(metrics) for metrics in metrics_list),
        dtype=float,
        count=len(metrics_list),
    )

    risk_mask = np.ones_like(metric_values, dtype=bool)
    if max_drawdown is not None:
        risk_mask &= drawdowns <= max_drawdown
    if var_limit is not None:
        risk_mask &= var95_values <= var_limit

    valid_mask = risk_mask & np.isfinite(metric_values)
    if np.any(valid_mask):
        candidate_scores = np.where(valid_mask, metric_values, np.nan)
        best_idx = int(np.nanargmax(candidate_scores))
        best_threshold = float(thresholds_eval[best_idx])
        best_metrics = metrics_list[best_idx]
    else:
        if max_drawdown is not None or var_limit is not None:
            limit_desc = []
            if max_drawdown is not None:
                limit_desc.append("max_drawdown")
            if var_limit is not None:
                limit_desc.append("var_95")
            raise ValueError(
                "No decision threshold satisfies risk limits: "
                + ", ".join(limit_desc)
            )
        best_idx = 0
        best_threshold = float(thresholds_eval[best_idx])
        best_metrics = metrics_list[best_idx]

    return best_threshold, best_metrics
def bootstrap_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    returns: np.ndarray,
    n_boot: int = 1000,
    *,
    n_jobs: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Estimate metric confidence intervals via bootstrapping.

    Parameters
    ----------
    y:
        Ground truth binary labels.
    probs:
        Predicted probabilities for the positive class.
    returns:
        Profit/return per sample used for Sharpe, Sortino and profit metrics.
    n_boot:
        Number of bootstrap resamples.
    n_jobs:
        Optional parallelism level for independent bootstrap resamples.  When
        ``None`` a sensible default is used.  Results are deterministic for any
        thread count because bootstrap indices are generated up-front.

    Returns
    -------
    dict
        Mapping of metric names to a dict with ``mean`` and 95%% confidence
        interval bounds ``low`` and ``high``.
    """

    if np is None:  # pragma: no cover - optional dependency guard
        raise ImportError("numpy is required for bootstrap metrics")

    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if y.size == 0 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(0)
    n = y.shape[0]
    keys = [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "var_95",
        "profit",
    ]

    index_samples = rng.integers(0, n, size=(n_boot, n))

    def _compute(sample_idx: "np.ndarray") -> "np.ndarray":
        metrics = _classification_metrics(
            y[sample_idx], probs[sample_idx], returns[sample_idx]
        )
        return np.asarray(
            [
                float(metrics.get(k)) if metrics.get(k) is not None else np.nan
                for k in keys
            ],
            dtype=float,
        )

    effective_jobs = _resolve_n_jobs(n_jobs)
    if effective_jobs == 1 or n_boot == 1 or not _HAS_JOBLIB:
        stacked = np.stack([_compute(idx) for idx in index_samples], axis=0)
    else:
        stacked = np.stack(
            Parallel(n_jobs=effective_jobs, prefer="threads")(
                delayed(_compute)(idx) for idx in index_samples
            ),
            axis=0,
        )

    valid_mask = ~np.all(np.isnan(stacked), axis=0)
    key_array = np.asarray(keys, dtype=object)
    results: Dict[str, Dict[str, float]] = {}
    if np.any(valid_mask):
        trimmed = stacked[:, valid_mask]
        means = np.nanmean(trimmed, axis=0)
        lows = np.nanquantile(trimmed, 0.025, axis=0)
        highs = np.nanquantile(trimmed, 0.975, axis=0)
        for key, mean, low, high in zip(key_array[valid_mask], means, lows, highs):
            results[key] = {
                "mean": float(mean),
                "low": float(low),
                "high": float(high),
            }
            logger.info(
                "%s: mean=%.6f, 95%% CI=(%.6f, %.6f)",
                key,
                results[key]["mean"],
                results[key]["low"],
                results[key]["high"],
            )

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, profits: np.ndarray | None = None
) -> Dict[str, object]:
    """Evaluate ``model`` on ``X`` and ``y`` returning rich metrics."""

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
        probas = preds.astype(float)
    return _classification_metrics(y, probas, profits)


def _abs_drawdown(returns: Sequence[float]) -> float:
    """Calculate absolute drawdown of cumulative returns.

    Parameters
    ----------
    returns:
        Sequence of per-trade or per-period returns.

    Returns
    -------
    float
        Maximum peak-to-trough decline in absolute terms.
    """

    if np is None:
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown
        return float(max_dd)

    arr = _as_float_array(returns)
    if arr.size == 0:
        return 0.0
    cumulative = np.concatenate(([0.0], np.cumsum(arr)))
    peaks = np.maximum.accumulate(cumulative)
    drawdowns = peaks - cumulative
    return float(np.max(drawdowns))


def _risk(returns: Sequence[float]) -> float:
    """Sample standard deviation of returns."""

    if np is None:
        returns = list(returns)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    arr = _as_float_array(returns)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _budget_utilisation(returns: Sequence[float], budget: float) -> float:
    """Return fraction of budget consumed by ``returns``.

    ``budget`` represents the allowed absolute sum of returns.  Values greater
    than ``1.0`` indicate that the budget has been exceeded.
    """

    if np is None:
        used = float(sum(abs(r) for r in returns))
    else:
        arr = _as_float_array(returns)
        used = float(np.abs(arr).sum())
    if budget <= 0:
        return used
    return used / float(budget)


def _order_type_compliance(order_types: Sequence[str], allowed: Sequence[str]) -> float:
    """Fraction of trades using allowed order types.

    Parameters
    ----------
    order_types:
        Recorded order types for a strategy.
    allowed:
        Order types permitted under the risk policy.
    """

    if not order_types:
        return 1.0
    if np is None:
        allowed_set = {o.lower() for o in allowed}
        cleaned = [o.strip().lower() for o in order_types]
        compliant = sum(1 for o in cleaned if o in allowed_set)
        return compliant / len(cleaned)

    cleaned = np.char.lower(
        np.char.strip(np.asarray(list(order_types), dtype=str))
    )
    allowed_arr = np.char.lower(np.asarray(list(allowed), dtype=str))
    if allowed_arr.size == 0:
        return 0.0 if cleaned.size else 1.0
    mask = np.isin(cleaned, allowed_arr)
    return float(mask.mean())


def _var_95(returns: Sequence[float]) -> float:
    """5%% value-at-risk of ``returns``.

    The implementation intentionally avoids heavy dependencies and simply
    computes the 5th percentile of the sorted return series.  An empty series
    yields ``0.0``.
    """

    if np is None:
        if not returns:
            return 0.0
        ordered = sorted(returns)
        idx = int(0.05 * len(ordered))
        idx = min(max(idx, 0), len(ordered) - 1)
        return float(ordered[idx])

    arr = _as_float_array(returns)
    if arr.size == 0:
        return 0.0
    ordered = np.sort(arr)
    idx = int(0.05 * ordered.size)
    idx = min(max(idx, 0), ordered.size - 1)
    return float(ordered[idx])


def _volatility_spikes(returns: Sequence[float]) -> int:
    """Count of returns deviating more than three standard deviations."""

    if np is None:
        if not returns:
            return 0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance)
        if std == 0:
            return 0
        return sum(1 for r in returns if abs(r - mean) > 3 * std)

    arr = _as_float_array(returns)
    if arr.size == 0:
        return 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std == 0.0:
        return 0
    deviations = np.abs(arr - mean) > (3.0 * std)
    return int(np.count_nonzero(deviations))


def _slippage_stats(slippage: Sequence[float]) -> tuple[float, float]:
    """Return mean and standard deviation of slippage values."""

    if np is None:
        slippage = list(slippage)
        if not slippage:
            return 0.0, 0.0
        mean = sum(slippage) / len(slippage)
        variance = sum((s - mean) ** 2 for s in slippage) / len(slippage)
        return mean, math.sqrt(variance)

    arr = _as_float_array(slippage)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    return mean, std


def evaluate_strategy(
    returns: Sequence[float],
    order_types: Sequence[str],
    slippage: Sequence[float] | None = None,
    *,
    budget: float,
    allowed_order_types: Sequence[str],
) -> Dict[str, float]:
    """Compute risk metrics for a trading strategy.

    This lightweight helper is used by the promotion tooling to ensure that
    candidate strategies obey basic risk constraints before being moved to the
    live directory.
    """

    returns_seq = list(returns)
    slip_seq = list(slippage or [])
    if np is not None:
        returns_data = np.asarray(returns_seq, dtype=float)
        slip_data = np.asarray(slip_seq, dtype=float)
    else:
        returns_data = returns_seq
        slip_data = slip_seq

    slip_mean, slip_std = _slippage_stats(slip_data)
    if np is not None and isinstance(returns_data, np.ndarray):
        mean = float(np.mean(returns_data)) if returns_data.size else 0.0
    else:
        mean = sum(returns_seq) / len(returns_seq) if returns_seq else 0.0

    risk = _risk(returns_data)
    if np is not None and isinstance(returns_data, np.ndarray):
        downside = returns_data[returns_data < 0]
        downside_std = (
            float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
        )
    else:
        downside = [r for r in returns_seq if r < 0]
        downside_std = (
            math.sqrt(sum(r * r for r in downside) / len(downside))
            if downside
            else 0.0
        )
    sharpe = mean / risk if risk > 0 else 0.0
    sortino = mean / downside_std if downside_std > 0 else 0.0
    drawdown = _abs_drawdown(returns_data)
    metrics = {
        "abs_drawdown": drawdown,
        "max_drawdown": drawdown,
        "risk": risk,
        "mean_return": mean,
        "budget_utilisation": _budget_utilisation(returns_data, budget),
        "order_type_compliance": _order_type_compliance(
            order_types, allowed_order_types
        ),
        "var_95": _var_95(returns_data),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "volatility_spikes": _volatility_spikes(returns_data),
        "slippage_mean": slip_mean,
        "slippage_std": slip_std,
    }
    stress_results = run_stress_tests(returns_seq, order_types=order_types)
    metrics["stress_tests"] = {
        name: asdict(result) for name, result in stress_results.items()
    }
    metrics["stress_summary"] = summarise_stress_results(stress_results)
    return metrics
