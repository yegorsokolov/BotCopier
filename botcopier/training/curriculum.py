"""Utilities for curriculum learning during training."""
from __future__ import annotations

import logging
from typing import Sequence, Tuple, List, Dict

import numpy as np

from botcopier.models.registry import get_model

logger = logging.getLogger(__name__)

SEQUENCE_MODELS = {"tabtransformer", "tcn", "crossmodal"}


def _apply_curriculum(
    X: np.ndarray,
    y: np.ndarray,
    profits: np.ndarray,
    sample_weight: np.ndarray,
    *,
    model_type: str,
    gpu_kwargs: dict[str, object],
    grad_clip: float,
    threshold: float,
    steps: int,
    R: np.ndarray | None = None,
    regime_feature_names: Sequence[str] | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    List[Dict[str, object]],
]:
    """Stage datasets from simple to complex and train sequentially.

    Returns filtered arrays and curriculum metadata.

    The dataset is ordered from "easy" to "hard" based on a simple
    volatility/complexity heuristic.  Samples with low feature volatility
    and small absolute profits are considered easier than highly volatile
    and high profit samples.  The model is trained sequentially on these
    partitions promoting to harder levels only if the validation profit
    exceeds ``threshold``.
    """
    if threshold <= 0.0 or steps <= 0 or not profits.size:
        return X, y, profits, sample_weight, R, []

    # Estimate a difficulty score for each sample using feature volatility
    # and profit magnitude.  ``np.std`` works with a single feature as well
    # but we clip to ``profits`` size for safety.
    if X.size:
        volatility = np.std(X, axis=1)
    else:
        volatility = np.zeros_like(profits)
    difficulty = volatility + np.abs(profits)
    order = np.argsort(difficulty)
    phases = np.array_split(order, steps)
    curriculum_meta: List[Dict[str, object]] = []
    final_idx = phases[0]

    for phase_num in range(1, steps + 1):
        idx_current = np.concatenate(phases[:phase_num])
        if len(idx_current) < 2:
            final_idx = idx_current
            break
        split = max(1, int(len(idx_current) * 0.8))
        tr_idx = idx_current[:split]
        val_idx = idx_current[split:]
        builder = get_model(model_type)
        kwargs = dict(**gpu_kwargs)
        if model_type == "moe" and R is not None and regime_feature_names is not None:
            model, pred_fn = builder(
                X[tr_idx],
                y[tr_idx],
                regime_features=R[tr_idx],
                regime_feature_names=regime_feature_names,
                sample_weight=sample_weight[tr_idx],
                grad_clip=grad_clip,
                **kwargs,
            )
            prob_val = pred_fn(X[val_idx], R[val_idx])
        else:
            if model_type not in SEQUENCE_MODELS:
                kwargs["sample_weight"] = sample_weight[tr_idx]
            if model_type == "moe" or model_type in SEQUENCE_MODELS:
                kwargs["grad_clip"] = grad_clip
            model, pred_fn = builder(X[tr_idx], y[tr_idx], **kwargs)
            prob_val = pred_fn(X[val_idx])
        val_profit = float(np.mean(profits[val_idx] * (prob_val >= 0.5)))
        curriculum_meta.append(
            {
                "phase": phase_num,
                "n_samples": int(len(idx_current)),
                "val_profit": val_profit,
            }
        )
        logger.info(
            "Curriculum phase %d: n=%d val_profit=%.6f",
            phase_num,
            len(idx_current),
            val_profit,
        )
        final_idx = idx_current
        if val_profit < threshold:
            break

    X_sel = X[final_idx]
    y_sel = y[final_idx]
    profits_sel = profits[final_idx]
    weight_sel = sample_weight[final_idx]
    R_sel = R[final_idx] if R is not None else None
    return X_sel, y_sel, profits_sel, weight_sel, R_sel, curriculum_meta
