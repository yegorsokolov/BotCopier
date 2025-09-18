#!/usr/bin/env python3
"""Fetch financial news and compute contextual sentiment embeddings.

This utility downloads recent headlines for given symbols and encodes them
using a modern :mod:`sentence_transformers` model.  The pooled embeddings are
stored in a SQLite and CSV file keeping only a rolling window of recent entries
per symbol.  The CSV output can be consumed during model training or by a
lightweight runtime service that provides the latest embeddings to trading
algorithms.

Example
-------
    # Run once for EURUSD and GBPUSD
    python scripts/fetch_news.py EURUSD GBPUSD

    # Update sentiment every 30 minutes
    python scripts/fetch_news.py EURUSD GBPUSD --interval 1800

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
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ENCODER = None


def _get_encoder():
    """Lazily construct the sentence transformer encoder."""

    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(_MODEL_NAME)
    return _ENCODER


def compute_embedding(headlines: Iterable[str]) -> tuple[np.ndarray, int]:
    """Return a pooled sentence embedding and headline count."""

    encoder = _get_encoder()
    texts = [str(h).strip() for h in headlines if isinstance(h, str) and h.strip()]
    dim = int(getattr(encoder, "get_sentence_embedding_dimension", lambda: 0)())
    if not texts:
        if dim <= 0:
            dim = 384  # fall back to the default dimension of MiniLM-L6
        return np.zeros(dim, dtype=float), 0
    emb = encoder.encode(
        texts,
        batch_size=min(len(texts), 32),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if emb.ndim == 1:
        pooled = emb.astype(float)
    else:
        pooled = emb.mean(axis=0).astype(float)
    if dim <= 0:
        dim = pooled.shape[-1]
    return pooled, len(texts)


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
        """
        CREATE TABLE IF NOT EXISTS sentiment (
            symbol TEXT,
            timestamp TEXT,
            embedding TEXT,
            dimension INTEGER,
            headline_count INTEGER
        )
        """
    )
    cur = conn.execute("PRAGMA table_info(sentiment)")
    cols = {row[1] for row in cur.fetchall()}
    if "embedding" not in cols:
        conn.execute("ALTER TABLE sentiment ADD COLUMN embedding TEXT")
    if "dimension" not in cols:
        conn.execute("ALTER TABLE sentiment ADD COLUMN dimension INTEGER DEFAULT 0")
    if "headline_count" not in cols:
        conn.execute(
            "ALTER TABLE sentiment ADD COLUMN headline_count INTEGER DEFAULT 0"
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
    for sym in symbols:
        headlines = fetch_headlines(sym)
        embedding, count = compute_embedding(headlines)
        dim = int(embedding.shape[0]) if embedding.size else 0
        emb_json = json.dumps(embedding.tolist())
        conn.execute(
            """
            INSERT INTO sentiment(symbol, timestamp, embedding, dimension, headline_count)
            VALUES(?,?,?,?,?)
            """,
            (sym, now, emb_json, dim, count),
        )
    _prune_old_entries(conn, window)
    conn.commit()
    # export to CSV for non-python consumers
    df = pd.read_sql_query(
        """
        SELECT symbol, timestamp, embedding, dimension, headline_count
        FROM sentiment
        ORDER BY timestamp
        """,
        conn,
    )
    conn.close()
    df = df.rename(
        columns={
            "timestamp": "sentiment_timestamp",
            "dimension": "sentiment_dimension",
            "headline_count": "sentiment_headline_count",
        }
    )
    if df.empty:
        df = df.drop(columns=["embedding"], errors="ignore")
        df.to_csv(csv_path, index=False)
        return

    def _expand(row: pd.Series) -> List[float]:
        emb_raw = row.get("embedding")
        try:
            values = json.loads(emb_raw) if isinstance(emb_raw, str) else []
        except json.JSONDecodeError:
            values = []
        dim_val = int(row.get("sentiment_dimension") or len(values))
        if dim_val and len(values) != dim_val:
            values = list(values)[:dim_val]
        return list(values)

    embeddings = df.apply(_expand, axis=1)
    max_dim = max((len(vec) for vec in embeddings), default=0)
    if max_dim:
        emb_mat = np.zeros((len(df), max_dim), dtype=float)
        for idx, vec in enumerate(embeddings):
            if not vec:
                continue
            arr = np.asarray(vec, dtype=float)
            length = min(len(arr), max_dim)
            emb_mat[idx, :length] = arr[:length]
        emb_cols = [f"sentiment_emb_{i}" for i in range(max_dim)]
        emb_df = pd.DataFrame(emb_mat, columns=emb_cols, index=df.index)
        df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
    else:
        df = df.drop(columns=["embedding"], errors="ignore")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news sentiment for symbols")
    parser.add_argument("symbols", nargs="+", help="Symbols to fetch, e.g. EURUSD")
    parser.add_argument("--db", type=Path, default=Path("news_sentiment.db"))
    parser.add_argument("--csv", type=Path, default=Path("news_sentiment.csv"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds between fetches; 0 to run once",
    )
    args = parser.parse_args()
    if not os.getenv("NEWSAPI_API_KEY"):
        print("NEWSAPI_API_KEY not set; skipping fetch")
    else:
        while True:
            update_store(args.symbols, args.db, args.csv, args.window)
            if args.interval <= 0:
                break
            time.sleep(args.interval)
