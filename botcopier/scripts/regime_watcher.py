#!/usr/bin/env python3
"""Watch market data and broadcast regime changes via shared memory ring.

This utility maintains an online clustering model (streaming k-means) over
volatility and spread features calculated from an incoming tick stream.  When
the predicted cluster changes it publishes the new regime identifier to a
shared memory ring buffer so that lightweight consumers (e.g. MQL experts) can
react without heavy dependencies.

Ticks are read from ``stdin`` as ``bid ask`` pairs separated by whitespace.  The
script is intentionally simple so that it can be integrated into a larger data
pipeline.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

import numpy as np
from river import cluster

try:  # pragma: no cover - used when packaged
    from .shm_ring import ShmRing
except Exception:  # pragma: no cover - fallback for direct execution
    from shm_ring import ShmRing

MSG_REGIME = 3  # keep consistent with MQL side


class RegimeWatcher:
    """Online clustering of market regimes."""

    def __init__(self, ring: ShmRing, clusters: int, window: int) -> None:
        self.ring = ring
        self.model = cluster.KMeans(n_clusters=clusters, halflife=0.5)
        self.prices: deque[float] = deque(maxlen=window)
        self.spreads: deque[float] = deque(maxlen=window)
        self.last_regime: int | None = None

    # ------------------------------------------------------------------
    def consume_tick(self, bid: float, ask: float) -> None:
        """Update statistics with a new tick and emit regime if changed."""
        price = (bid + ask) / 2.0
        spread = ask - bid
        self.prices.append(price)
        self.spreads.append(spread)
        if len(self.prices) < self.prices.maxlen:
            return  # wait for enough history
        vol = float(np.std(self.prices))
        avg_spread = float(np.mean(self.spreads))
        x = {"vol": vol, "spread": avg_spread}
        self.model = self.model.learn_one(x)
        regime = int(self.model.predict_one(x))
        if regime != self.last_regime:
            self.ring.push(MSG_REGIME, bytes([regime]))
            self.last_regime = regime


# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Stream clustering of regimes")
    p.add_argument("--ring-path", default="/tmp/regime_ring", help="path to shared memory ring")
    p.add_argument("--clusters", type=int, default=3, help="number of regimes")
    p.add_argument("--window", type=int, default=100, help="window length for stats")
    args = p.parse_args()

    ring = ShmRing.create(args.ring_path, 1 << 10)
    watcher = RegimeWatcher(ring, clusters=args.clusters, window=args.window)

    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        bid, ask = map(float, parts[:2])
        watcher.consume_tick(bid, ask)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
