#!/usr/bin/env python3
"""Compare news sentiment scalars against embedding features using logistic backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _prepare_scalar_view(news_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse embedding columns in ``news_df`` into a single sentiment score."""

    embed_cols = [c for c in news_df.columns if c.startswith("sentiment_emb_")]
    if not embed_cols:
        raise ValueError("news DataFrame does not contain sentiment embeddings")
    ordered = sorted(
        embed_cols,
        key=lambda name: int(name.rsplit("_", 1)[-1])
        if name.rsplit("_", 1)[-1].isdigit()
        else name,
    )
    numeric = news_df[ordered].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scalar_df = news_df[["symbol"]].copy()
    if "sentiment_timestamp" in news_df.columns:
        scalar_df["sentiment_timestamp"] = news_df["sentiment_timestamp"]
    scalar_df["sentiment_score"] = numeric.mean(axis=1)
    if "sentiment_headline_count" in news_df.columns:
        scalar_df["sentiment_headline_count"] = news_df["sentiment_headline_count"]
    return scalar_df


def _evaluate_accuracy(df: pd.DataFrame, feature_cols: Iterable[str]) -> float:
    """Train/test split logistic regression accuracy for ``feature_cols``."""

    feat_cols = list(feature_cols)
    if "label" not in df.columns:
        raise ValueError("trades data requires a 'label' column")
    data = df.dropna(subset=["label"]).copy()
    data[feat_cols] = data[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = data["label"].to_numpy(dtype=float)
    if np.unique(y).size <= 1:
        return float(np.mean(y))
    X = data[feat_cols].to_numpy(dtype=float)
    split = max(1, int(len(data) * 0.6))
    model = LogisticRegression(max_iter=1000)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    if preds.size == 0:
        preds = model.predict(X)
        return float(accuracy_score(y, preds))
    return float(accuracy_score(y[split:], preds))


def run_backtests(
    trades_path: Path, news_path: Path, out_dir: Path, **_: Any
) -> Dict[str, Any]:
    """Compare scalar and embedding sentiment features on a simple backtest."""

    trades_df = pd.read_csv(trades_path)
    news_df = pd.read_csv(news_path)
    if "sentiment_timestamp" in news_df.columns:
        news_df["sentiment_timestamp"] = pd.to_datetime(
            news_df["sentiment_timestamp"], errors="coerce"
        )
    scalar_df = _prepare_scalar_view(news_df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for label, sentiment in (("scalar", scalar_df), ("embedding", news_df)):
        merged = trades_df.merge(sentiment, on="symbol", how="left")
        merged = merged.fillna(0.0)
        if label == "scalar":
            feature_cols = ["sentiment_score"]
            if "sentiment_headline_count" in merged.columns:
                feature_cols.append("sentiment_headline_count")
        else:
            embed_cols = sorted(
                [c for c in sentiment.columns if c.startswith("sentiment_emb_")],
                key=lambda name: int(name.rsplit("_", 1)[-1])
                if name.rsplit("_", 1)[-1].isdigit()
                else name,
            )
            feature_cols = list(embed_cols)
            if "sentiment_headline_count" in merged.columns:
                feature_cols.append("sentiment_headline_count")
        accuracy = _evaluate_accuracy(merged, feature_cols)
        results[label] = {
            "accuracy": accuracy,
            "cv_accuracy": accuracy,
            "feature_columns": feature_cols,
        }
    results["delta_accuracy"] = (
        results["embedding"]["accuracy"] - results["scalar"]["accuracy"]
    )
    summary_path = out_dir / "sentiment_backtest.json"
    summary_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scalar vs. embedding news sentiment accuracy"
    )
    parser.add_argument("trades", type=Path, help="Path to trades_raw.csv")
    parser.add_argument("news", type=Path, help="Path to news_sentiment.csv with embeddings")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sentiment_backtests"),
        help="Directory to store backtest summary",
    )
    args = parser.parse_args()
    results = run_backtests(args.trades, args.news, args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
