"""Portfolio allocation utilities."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _cluster_variance(cov: pd.DataFrame, indices: Sequence[int]) -> float:
    """Compute variance for a cluster defined by ``indices``.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of asset returns.
    indices : Sequence[int]
        Column indices corresponding to the assets in the cluster.

    Returns
    -------
    float
        The variance of the minimum-variance portfolio of the cluster.
    """

    sub = cov.iloc[indices, indices].to_numpy()
    inv_diag = 1.0 / np.diag(sub)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def hierarchical_risk_parity(returns: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Compute Hierarchical Risk Parity (HRP) portfolio weights.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns with one column per asset.

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        Series of weights indexed by asset name and the linkage matrix
        describing the clustering dendrogram.
    """

    if returns.shape[1] == 1:
        w = pd.Series([1.0], index=returns.columns)
        return w, np.zeros((0, 4))

    cov = returns.cov().fillna(0.0)
    corr = returns.corr().fillna(0.0)
    dist = np.sqrt(0.5 * (1 - corr))
    condensed = squareform(dist.to_numpy(), checks=False)
    link = linkage(condensed, method="single")
    order = leaves_list(link)
    ordered_cols = returns.columns[order]
    cov = cov.loc[ordered_cols, ordered_cols]

    weights = pd.Series(1.0, index=ordered_cols)
    clusters: list[list[int]] = [list(range(len(ordered_cols)))]

    while clusters:
        idx = clusters.pop(0)
        if len(idx) <= 1:
            continue
        split = len(idx) // 2
        left = idx[:split]
        right = idx[split:]
        var_left = _cluster_variance(cov, left)
        var_right = _cluster_variance(cov, right)
        alloc_left = 1.0 - var_left / (var_left + var_right)
        alloc_right = 1.0 - alloc_left
        weights.iloc[left] *= alloc_left
        weights.iloc[right] *= alloc_right
        clusters.append(left)
        clusters.append(right)

    weights /= weights.sum()
    weights = weights.reindex(returns.columns).fillna(0.0)
    return weights, link
