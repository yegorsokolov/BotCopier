from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence

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
from botcopier.utils.validation import validate_columns
from metrics.registry import get_metrics, register_metric
from schemas.decisions import DECISION_SCHEMA
from schemas.trades import TRADE_SCHEMA

logger = logging.getLogger(__name__)


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
    except OSError as exc:
        err = DataError(
            f"Failed to read predictions {pred_file}", timestamp=datetime.now(UTC)
        )
        logger.error("%s; using empty DataFrame", err, exc_info=exc)
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
    except OSError as exc:
        err = DataError(
            f"Failed to read trade log {log_file}", timestamp=datetime.now(UTC)
        )
        logger.error("%s; using empty DataFrame", err, exc_info=exc)
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

    matched_mask = merged["ticket"].notna()
    matches = int(matched_mask.sum())
    profits = merged.loc[matched_mask, "profit"].astype(float)
    net_profits = profits - (fee_per_trade + np.abs(profits) * slippage_bps * 1e-4)
    trade_times = merged.loc[matched_mask, "close_time"]

    gross_profit = float(profits[profits >= 0].sum())
    gross_loss = float(-profits[profits < 0].sum())

    predictions_per_model = (
        merged["executed_model_idx"].dropna().astype(int).value_counts().to_dict()
    )
    matches_per_model = (
        merged.loc[matched_mask, "executed_model_idx"]
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )

    mask_probs = merged["probability"].notna()
    y_true = matched_mask.astype(int)
    y_score = merged["probability"].fillna(0.0)
    y_true_list = y_true[mask_probs].tolist()
    y_score_list = y_score[mask_probs].tolist()
    y_true_list.extend([1] * len(unused_trades))
    y_score_list.extend([0.0] * len(unused_trades))

    bound_in = 0
    bound_total = 0
    if conformal_lower is not None and conformal_upper is not None:
        bound_total = int(mask_probs.sum())
        bound_in = int(
            (
                (merged.loc[mask_probs, "probability"] >= conformal_lower)
                & (merged.loc[mask_probs, "probability"] <= conformal_upper)
            ).sum()
        )

    mu = merged.loc[matched_mask, "value"]
    log_v = merged.loc[matched_mask, "log_variance"]
    valid = mu.notna() & log_v.notna()
    var = np.exp(log_v[valid])
    p = profits[valid]
    nll_values = 0.5 * (np.log(2 * np.pi) + log_v[valid] + ((p - mu[valid]) ** 2) / var)
    nd = NormalDist()
    z = nd.inv_cdf(0.05)
    c = nd.pdf(z) / 0.05
    es_pred_values = mu[valid] - np.sqrt(var) * c

    tp = matches
    fp = len(predictions) - matches
    fn = len(trades) - matches
    total = tp + fp + fn
    accuracy = tp / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy = (gross_profit - gross_loss) / matches if matches else 0.0
    expected_return = float(profits.mean()) if matches else 0.0
    expected_return_net = float(net_profits.mean()) if matches else 0.0
    downside = profits[profits < 0]
    downside_net = net_profits[net_profits < 0]
    downside_risk = float(-downside.mean()) if len(downside) else 0.0
    risk_reward = expected_return - downside_risk

    if matches:
        sorted_profits = np.sort(profits.values)
        n_tail = max(1, int(np.ceil(len(sorted_profits) * 0.05)))
        tail = sorted_profits[:n_tail]
        cvar = float(tail.mean())
        var_95 = float(np.quantile(profits, 0.05))
        es_95 = float(profits[profits <= var_95].mean())
    else:
        cvar = var_95 = es_95 = 0.0

    # sharpe and precision metrics are handled via evaluation hooks

    roc_auc = pr_auc = brier = ece = None
    reliability = {"prob_true": [], "prob_pred": []}
    if y_score_list:
        brier = brier_score_loss(y_true_list, y_score_list)
        try:
            prob_true, prob_pred = calibration_curve(
                y_true_list, y_score_list, n_bins=10
            )
            reliability = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            }
            bins = np.linspace(0.0, 1.0, 11)
            binids = np.digitize(y_score_list, bins[1:-1], right=True)
            bin_counts = np.bincount(binids, minlength=10)
            counts = bin_counts[bin_counts > 0]
            ece = float(
                np.sum(np.abs(prob_true - prob_pred) * (counts / len(y_score_list)))
            )
        except ValueError:
            pass
        if len(set(y_true_list)) > 1:
            roc_auc = roc_auc_score(y_true_list, y_score_list)
            pr_auc = average_precision_score(y_true_list, y_score_list)

    conformal = bound_in / bound_total if bound_total else None
    nll_mean = float(nll_values.mean()) if len(nll_values) else None
    es_pred_mean = float(es_pred_values.mean()) if len(es_pred_values) else None

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
        "cvar": cvar,
        "var_95": var_95,
        "es_95": es_95,
        "risk_reward": risk_reward,
        "conformal_coverage": conformal,
        "predictions_per_model": predictions_per_model,
        "matches_per_model": matches_per_model,
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
    from botcopier.eval.hooks import dispatch_hooks

    dispatch_hooks(hooks, ctx)

    return ctx["stats"]


def _metric_accuracy(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    preds = (probas >= 0.5).astype(int)
    return float(accuracy_score(y_true, preds))


def _metric_roc_auc(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float | None:
    return (
        float(roc_auc_score(y_true, probas)) if len(set(y_true.tolist())) > 1 else None
    )


def _metric_pr_auc(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float | None:
    return (
        float(average_precision_score(y_true, probas))
        if len(set(y_true.tolist())) > 1
        else None
    )


def _metric_brier(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    return float(brier_score_loss(y_true, probas))


def _metric_reliability(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> Dict[str, List[float]]:
    try:
        prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10)
        return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    except ValueError:
        return {"prob_true": [], "prob_pred": []}


def _metric_ece(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
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
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None
) -> np.ndarray:
    preds = (probas >= 0.5).astype(int)
    if profits is None:
        return np.where(preds == y_true, 1.0, -1.0)
    return profits


def _metric_sharpe(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    returns = _returns(y_true, probas, profits)
    if returns.size > 1:
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        if std > 0:
            return mean / std
    return 0.0


def _metric_sortino(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    returns = _returns(y_true, probas, profits)
    if returns.size > 0:
        mean = float(np.mean(returns))
        downside = returns[returns < 0]
        if downside.size > 0:
            downside_std = float(np.sqrt(np.mean(downside**2)))
            if downside_std > 0:
                return mean / downside_std
    return 0.0


def _metric_drawdown(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    returns = _returns(y_true, probas, profits).tolist()
    return float(_abs_drawdown(returns))


def _metric_var95(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    returns = _returns(y_true, probas, profits).tolist()
    return float(_var_95(returns))


def _metric_profit(
    y_true: np.ndarray, probas: np.ndarray, profits: np.ndarray | None = None
) -> float:
    returns = _returns(y_true, probas, profits)
    return float(np.sum(returns))


# Register default metrics
register_metric("accuracy", _metric_accuracy)
register_metric("roc_auc", _metric_roc_auc)
register_metric("pr_auc", _metric_pr_auc)
register_metric("brier_score", _metric_brier)
register_metric("reliability_curve", _metric_reliability)
register_metric("ece", _metric_ece)
register_metric("sharpe_ratio", _metric_sharpe)
register_metric("sortino_ratio", _metric_sortino)
register_metric("max_drawdown", _metric_drawdown)
register_metric("var_95", _metric_var95)
register_metric("profit", _metric_profit)


def _classification_metrics(
    y_true: np.ndarray,
    probas: np.ndarray,
    profits: np.ndarray | None = None,
    selected: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Compute classification and calibration metrics."""

    results: Dict[str, object] = {}
    for name, fn in get_metrics(selected).items():
        try:
            results[name] = fn(y_true, probas, profits)
        except (
            Exception
        ) as exc:  # pragma: no cover - specific errors tested via injection
            err = ServiceError(f"Metric {name} failed", timestamp=datetime.now(UTC))
            logger.warning("%s", err, exc_info=exc)
            results[name] = np.nan
    return results


def bootstrap_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    returns: np.ndarray,
    n_boot: int = 1000,
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

    Returns
    -------
    dict
        Mapping of metric names to a dict with ``mean`` and 95%% confidence
        interval bounds ``low`` and ``high``.
    """

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
    collected: Dict[str, List[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        metrics = _classification_metrics(y[idx], probs[idx], returns[idx])
        for k in keys:
            v = metrics.get(k)
            if v is not None:
                collected[k].append(float(v))

    results: Dict[str, Dict[str, float]] = {}
    for k, vals in collected.items():
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        low, high = np.quantile(arr, [0.025, 0.975])
        results[k] = {"mean": mean, "low": float(low), "high": float(high)}

    for k, v in results.items():
        logger.info(
            "%s: mean=%.6f, 95%% CI=(%.6f, %.6f)", k, v["mean"], v["low"], v["high"]
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


def _risk(returns: Sequence[float]) -> float:
    """Sample standard deviation of returns."""

    returns = list(returns)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _budget_utilisation(returns: Sequence[float], budget: float) -> float:
    """Return fraction of budget consumed by ``returns``.

    ``budget`` represents the allowed absolute sum of returns.  Values greater
    than ``1.0`` indicate that the budget has been exceeded.
    """

    used = float(sum(abs(r) for r in returns))
    if budget <= 0:
        return used
    return used / budget


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
    allowed_set = {o.lower() for o in allowed}
    cleaned = [o.strip().lower() for o in order_types]
    compliant = sum(1 for o in cleaned if o in allowed_set)
    return compliant / len(cleaned)


def _var_95(returns: Sequence[float]) -> float:
    """5%% value-at-risk of ``returns``.

    The implementation intentionally avoids heavy dependencies and simply
    computes the 5th percentile of the sorted return series.  An empty series
    yields ``0.0``.
    """

    if not returns:
        return 0.0
    ordered = sorted(returns)
    idx = int(0.05 * len(ordered))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _volatility_spikes(returns: Sequence[float]) -> int:
    """Count of returns deviating more than three standard deviations."""

    if not returns:
        return 0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)
    if std == 0:
        return 0
    return sum(1 for r in returns if abs(r - mean) > 3 * std)


def _slippage_stats(slippage: Sequence[float]) -> tuple[float, float]:
    """Return mean and standard deviation of slippage values."""

    slippage = list(slippage)
    if not slippage:
        return 0.0, 0.0
    mean = sum(slippage) / len(slippage)
    variance = sum((s - mean) ** 2 for s in slippage) / len(slippage)
    return mean, math.sqrt(variance)


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

    returns = list(returns)
    slip = list(slippage or [])
    slip_mean, slip_std = _slippage_stats(slip)
    mean = sum(returns) / len(returns) if returns else 0.0
    risk = _risk(returns)
    downside = [r for r in returns if r < 0]
    if downside:
        downside_std = math.sqrt(sum(r * r for r in downside) / len(downside))
    else:
        downside_std = 0.0
    sharpe = mean / risk if risk > 0 else 0.0
    sortino = mean / downside_std if downside_std > 0 else 0.0
    drawdown = _abs_drawdown(returns)
    metrics = {
        "abs_drawdown": drawdown,
        "max_drawdown": drawdown,
        "risk": risk,
        "mean_return": mean,
        "budget_utilisation": _budget_utilisation(returns, budget),
        "order_type_compliance": _order_type_compliance(
            order_types, allowed_order_types
        ),
        "var_95": _var_95(returns),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "volatility_spikes": _volatility_spikes(returns),
        "slippage_mean": slip_mean,
        "slippage_std": slip_std,
    }
    return metrics
