import json
import logging
import math
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from botcopier.exceptions import DataError
from metrics.registry import get_metrics, register_metric

logger = logging.getLogger(__name__)


def _parse_time(value: str, *, symbol: str = "") -> datetime:
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise DataError("unrecognised time format", symbol=symbol, timestamp=value)


def _load_predictions(pred_file: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_file, delimiter=";")

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
    df = pd.read_csv(log_file, delimiter=";")

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
            raise DataError(
                "failed to read model metadata",
                symbol="model",
                timestamp=datetime.utcnow(),
            ) from exc

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
    net_profits = profits - (
        fee_per_trade + np.abs(profits) * slippage_bps * 1e-4
    )
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
    precision = tp / (tp + fp) if (tp + fp) else 0.0
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

    sharpe = sortino = 0.0
    sharpe_net = sortino_net = 0.0
    if matches > 1:
        mean = expected_return
        variance = float(profits.var(ddof=1))
        std = math.sqrt(variance)
        if std > 0:
            sharpe = mean / std
        if len(downside):
            downside_dev = math.sqrt(float((downside**2).mean()))
            if downside_dev > 0:
                sortino = mean / downside_dev
        mean_net = expected_return_net
        variance_net = float(net_profits.var(ddof=1))
        std_net = math.sqrt(variance_net)
        if std_net > 0:
            sharpe_net = mean_net / std_net
        if len(downside_net):
            downside_dev_net = math.sqrt(float((downside_net**2).mean()))
            if downside_dev_net > 0:
                sortino_net = mean_net / downside_dev_net

    annual_sharpe = annual_sortino = 0.0
    annual_sharpe_net = annual_sortino_net = 0.0
    if matches > 1 and not trade_times.isna().all():
        start = trade_times.min()
        end = trade_times.max()
        years = (end - start).total_seconds() / (365 * 24 * 3600)
        if years <= 0:
            years = 1.0
        trades_per_year = matches / years
        factor = math.sqrt(trades_per_year)
        annual_sharpe = sharpe * factor
        annual_sortino = sortino * factor
        annual_sharpe_net = sharpe_net * factor
        annual_sortino_net = sortino_net * factor

    roc_auc = pr_auc = brier = None
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
        "precision": precision,
        "recall": recall,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy": expectancy,
        "expected_return": expected_return,
        "downside_risk": downside_risk,
        "cvar": cvar,
        "var_95": var_95,
        "es_95": es_95,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "sharpe_ratio_annualised": annual_sharpe,
        "sortino_ratio_annualised": annual_sortino,
        "sharpe_ratio_net": sharpe_net,
        "sortino_ratio_net": sortino_net,
        "sharpe_ratio_net_annualised": annual_sharpe_net,
        "sortino_ratio_net_annualised": annual_sortino_net,
        "risk_reward": risk_reward,
        "conformal_coverage": conformal,
        "predictions_per_model": predictions_per_model,
        "matches_per_model": matches_per_model,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
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
    return stats


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
register_metric("sharpe_ratio", _metric_sharpe)
register_metric("sortino_ratio", _metric_sortino)
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
        results[name] = fn(y_true, probas, profits)
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
