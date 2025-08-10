#!/usr/bin/env python3
"""Fetch financial news and compute sentiment scores.

This utility downloads recent headlines for given symbols and scores them using
FinBERT from the :mod:`transformers` library.  Scores are stored in a SQLite
and CSV file keeping only a rolling window of recent entries per symbol.

Example
-------
    python scripts/fetch_news.py EURUSD GBPUSD

Environment variables
---------------------
``NEWSAPI_API_KEY`` may be set to provide an API key for the NewsAPI.org
service.  When no key is provided, the script simply exits without updating
any data.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
import pandas as pd


_MODEL_NAME = "yiyanghkust/finbert-tone"
_PIPELINE = None


def _get_pipeline():
    """Lazily construct the FinBERT sentiment pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _PIPELINE = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return _PIPELINE


def compute_sentiment(headlines: Iterable[str]) -> float:
    """Return average FinBERT sentiment score for ``headlines``.

    Positive scores indicate optimistic sentiment, negative scores indicate
    pessimism.  Neutral headlines contribute ``0``.
    """
    nlp = _get_pipeline()
    scores: List[float] = []
    for hl in headlines:
        if not hl:
            continue
        result = nlp(hl)[0]
        label = result.get("label", "").lower()
        score = float(result.get("score", 0.0))
        if label.startswith("positive"):
            scores.append(score)
        elif label.startswith("negative"):
            scores.append(-score)
        else:
            scores.append(0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def fetch_headlines(symbol: str) -> List[str]:
    """Fetch recent headlines for ``symbol`` using NewsAPI."""
    api_key = os.getenv("NEWSAPI_API_KEY")
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": symbol, "apiKey": api_key, "sortBy": "publishedAt"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [a.get("title", "") for a in data.get("articles", [])]
    except Exception:
        return []


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sentiment (symbol TEXT, timestamp TEXT, score REAL)"
    )


def _prune_old_entries(conn: sqlite3.Connection, window: int) -> None:
    cur = conn.cursor()
    syms = [row[0] for row in cur.execute("SELECT DISTINCT symbol FROM sentiment").fetchall()]
    for sym in syms:
        cur.execute(
            "DELETE FROM sentiment WHERE rowid NOT IN ("  # keep most recent ``window`` rows
            "SELECT rowid FROM sentiment WHERE symbol=? ORDER BY timestamp DESC LIMIT ?)",
            (sym, window),
        )
    conn.commit()


def update_store(
    symbols: Iterable[str],
    db_path: Path,
    csv_path: Path,
    window: int = 50,
) -> None:
    """Fetch headlines for ``symbols`` and update sentiment store."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    now = datetime.now(timezone.utc).isoformat()
    rows: List[Tuple[str, str, float]] = []
    for sym in symbols:
        headlines = fetch_headlines(sym)
        score = compute_sentiment(headlines)
        conn.execute(
            "INSERT INTO sentiment(symbol, timestamp, score) VALUES(?,?,?)",
            (sym, now, score),
        )
        rows.append((sym, now, score))
    _prune_old_entries(conn, window)
    conn.commit()
    # export to CSV for non-python consumers
    df = pd.read_sql_query(
        "SELECT symbol, timestamp, score FROM sentiment ORDER BY timestamp",
        conn,
    )
    conn.close()
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news sentiment for symbols")
    parser.add_argument("symbols", nargs="+", help="Symbols to fetch, e.g. EURUSD")
    parser.add_argument("--db", type=Path, default=Path("news_sentiment.db"))
    parser.add_argument("--csv", type=Path, default=Path("news_sentiment.csv"))
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if not os.getenv("NEWSAPI_API_KEY"):
        print("NEWSAPI_API_KEY not set; skipping fetch")
    else:
        update_store(args.symbols, args.db, args.csv, args.window)
