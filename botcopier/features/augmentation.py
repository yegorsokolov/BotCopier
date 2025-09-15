"""Data augmentation helpers for BotCopier."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .registry import register_feature


@register_feature("augment_dataframe")
def _augment_dataframe_impl(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Return DataFrame with additional augmented rows using mixup and jitter."""
    if ratio <= 0 or df.empty:
        return df

    n = len(df)
    n_aug = max(1, int(n * ratio))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[num_cols].std().replace(0, 1).to_numpy()

    aug_rows: list[pd.Series] = []
    for _ in range(n_aug):
        i1, i2 = np.random.randint(0, n, size=2)
        lam = np.random.beta(0.4, 0.4)
        row1 = df.iloc[i1]
        row2 = df.iloc[i2]
        new_row = row1.copy()
        if num_cols:
            mix = lam * row1[num_cols].to_numpy(dtype=float) + (1 - lam) * row2[
                num_cols
            ].to_numpy(dtype=float)
            jitter = np.random.normal(0.0, 0.01, size=len(num_cols)) * stats
            new_row[num_cols] = mix + jitter
        if "event_time" in df.columns and pd.notnull(new_row.get("event_time")):
            delta = np.random.uniform(-60, 60)
            new_row["event_time"] = new_row["event_time"] + pd.to_timedelta(
                delta, unit="s"
            )
        aug_rows.append(new_row)

    aug_df = pd.DataFrame(aug_rows)
    logging.info(
        "Augmenting data with %d synthetic rows (ratio %.3f)", n_aug, n_aug / n
    )
    return pd.concat([df, aug_df], ignore_index=True)


def _dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[list[tuple[int, int]], float]:
    """Return optimal DTW alignment path between two sequences and its cost."""
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path, float(dp[n, m])


@register_feature("augment_dtw_dataframe")
def _augment_dtw_dataframe_impl(
    df: pd.DataFrame, ratio: float, window: int = 3
) -> pd.DataFrame:
    """Return DataFrame augmented by DTW-based sequence mixup."""
    if ratio <= 0 or len(df) < 2:
        return df

    n = len(df)
    n_aug = max(1, int(n * ratio))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df

    windows: list[np.ndarray] = []
    for start in range(0, n - window + 1):
        windows.append(df.iloc[start : start + window][num_cols].to_numpy(dtype=float))
    if not windows:
        windows.append(df[num_cols].to_numpy(dtype=float))
        window = len(df)

    aug_rows: list[pd.Series] = []
    while len(aug_rows) < n_aug:
        i1 = np.random.randint(0, len(windows))
        seq1 = windows[i1]
        best_j = None
        best_dist = np.inf
        for j in range(len(windows)):
            if j == i1:
                continue
            _, dist = _dtw_path(seq1, windows[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None:
            break
        seq2 = windows[best_j]
        path, _ = _dtw_path(seq1, seq2)
        lam = np.random.beta(0.4, 0.4)
        for idx1, idx2 in path:
            row1 = df.iloc[i1 + idx1]
            row2 = df.iloc[best_j + idx2]
            new_row = row1.copy()
            new_row[num_cols] = lam * row1[num_cols].to_numpy(dtype=float) + (
                1 - lam
            ) * row2[num_cols].to_numpy(dtype=float)
            if "event_time" in df.columns:
                t1 = pd.to_datetime(row1.get("event_time"))
                t2 = pd.to_datetime(row2.get("event_time"))
                if pd.notnull(t1) and pd.notnull(t2):
                    new_row["event_time"] = t1 + (t2 - t1) * (1 - lam)
            new_row["aug_ratio"] = lam
            aug_rows.append(new_row)
            if len(aug_rows) >= n_aug:
                break

    if not aug_rows:
        return df
    aug_df = pd.DataFrame(aug_rows)
    logging.info(
        "DTW augmenting data with %d synthetic rows (ratio %.3f)",
        len(aug_df),
        len(aug_df) / n,
    )
    return pd.concat([df, aug_df], ignore_index=True)


# Public API functions; may be wrapped by ``configure_cache``
_augment_dataframe = _augment_dataframe_impl
_augment_dtw_dataframe = _augment_dtw_dataframe_impl
