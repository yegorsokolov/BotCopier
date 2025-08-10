#!/usr/bin/env python3
"""Online trainer that updates a River model from streaming events."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional

from river.linear_model import LogisticRegression
from river.tree import HoeffdingTreeClassifier
from river import stats

# optional imports for gRPC message definitions
try:
    from proto import observer_pb2, observer_pb2_grpc  # type: ignore
except Exception:  # pragma: no cover - proto generation not always available
    observer_pb2 = observer_pb2_grpc = None


class OnlineTrainer:
    """Incrementally train a model from streamed events."""

    def __init__(
        self,
        model_type: str = "logistic",
        save_path: Path | str = Path("model_online.json"),
        save_interval: int = 5,
    ) -> None:
        self.model_name = model_type
        if model_type == "logistic":
            self.model = LogisticRegression()
        else:
            self.model = HoeffdingTreeClassifier()
        self.save_path = Path(save_path)
        self.save_interval = save_interval * 60  # minutes -> seconds
        self.last_save = time.time()
        self.last_event_id = 0
        self.feature_stats: Dict[str, stats.Mean] = defaultdict(stats.Mean)
        if self.save_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
        except Exception:
            return
        self.last_event_id = data.get("last_event_id", 0)
        for f, info in data.get("feature_stats", {}).items():
            m = stats.Mean()
            m.n = info.get("count", 0)
            m.mean = info.get("mean", 0.0)
            self.feature_stats[f] = m
        if data.get("model_type") == "logistic":
            self.model = LogisticRegression()
            weights = data.get("coefficients", [])
            names = data.get("feature_names", [])
            for n, w in zip(names, weights):
                self.model._weights[n] = w
            self.model.intercept = data.get("intercept", 0.0)
            self.model_name = "logistic"
        elif data.get("model_type") == "hoeffding_tree":
            self.model = HoeffdingTreeClassifier()
            # Minimal restoration: Hoeffding trees in River expose ``__setstate__``
            # for pickled dictionaries.  Persist ``model_state`` if present.
            state = data.get("model_state")
            if state is not None:
                self.model.__setstate__(state)
            self.model_name = "hoeffding_tree"

    def save_model(self) -> None:
        feature_names = sorted(self.feature_stats.keys())
        payload: Dict[str, Any] = {
            "model_type": self.model_name,
            "feature_names": feature_names,
            "last_event_id": self.last_event_id,
            "feature_stats": {
                f: {"count": st.n, "mean": st.get()} for f, st in self.feature_stats.items()
            },
        }
        if isinstance(self.model, LogisticRegression):
            payload["coefficients"] = [self.model._weights.get(f, 0.0) for f in feature_names]
            payload["intercept"] = self.model.intercept
        elif isinstance(self.model, HoeffdingTreeClassifier):
            payload["model_state"] = self.model.__getstate__()
        with open(self.save_path, "w") as f:
            json.dump(payload, f)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------
    def process_event(self, event: Dict[str, Any]) -> None:
        features = event.get("features", {})
        target = event.get("y")
        event_id = event.get("event_id")
        if event_id is not None:
            self.last_event_id = int(event_id)
        if features and target is not None:
            self.model.learn_one(features, target)
            for k, v in features.items():
                self.feature_stats[k].update(v)
        now = time.time()
        if now - self.last_save >= self.save_interval:
            self.save_model()
            self.last_save = now

    # ------------------------------------------------------------------
    # Streaming interfaces
    # ------------------------------------------------------------------
    async def consume_websocket(self, url: str) -> None:
        import aiohttp  # imported lazily to avoid mandatory dependency

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(msg.data)
                            self.process_event(event)
                        except Exception:
                            continue

    async def consume_grpc(self, target: str) -> None:
        import grpc  # imported lazily

        if observer_pb2 is None:
            raise RuntimeError("gRPC proto definitions are not available")
        async with grpc.aio.insecure_channel(target) as channel:
            stub = observer_pb2_grpc.ObserverStub(channel)  # type: ignore
            request = observer_pb2.SubscribeRequest()  # type: ignore
            async for msg in stub.Subscribe(request):  # type: ignore
                event = {
                    "event_id": msg.event.event_id,
                    "features": dict(msg.event.features),
                    "y": msg.event.label,
                }
                self.process_event(event)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Online training from decision streams")
    p.add_argument("--ws-url")
    p.add_argument("--grpc-url")
    p.add_argument("--model", choices=["logistic", "hoeffding_tree"], default="logistic")
    p.add_argument("--save-path", default="model_online.json")
    p.add_argument("--save-interval", type=int, default=5, help="Minutes between checkpoint saves")
    args = p.parse_args(argv)
    trainer = OnlineTrainer(args.model, Path(args.save_path), args.save_interval)
    if args.ws_url:
        asyncio.run(trainer.consume_websocket(args.ws_url))
    elif args.grpc_url:
        asyncio.run(trainer.consume_grpc(args.grpc_url))
    else:
        # No streaming source specified; run a dummy loop waiting for events
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
