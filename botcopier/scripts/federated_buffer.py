#!/usr/bin/env python3
"""Federated experience buffer using gRPC.

This module provides a simple gRPC server that aggregates compressed
experience batches uploaded by multiple clients and returns a noisy
aggregate to preserve privacy.  Clients can upload batches and download
merged experiences for further training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
from concurrent import futures
from typing import List, Sequence, Tuple

import grpc
import numpy as np
from google.protobuf import empty_pb2

from botcopier.utils.random import set_seed
from logging_utils import setup_logging
from proto import federated_buffer_pb2 as pb2
from proto import federated_buffer_pb2_grpc as pb2_grpc

Experience = Tuple[np.ndarray, int, float, np.ndarray]


class _BufferServicer(pb2_grpc.ExperienceBufferServicer):
    """gRPC servicer implementing the federated buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._experiences: List[dict] = []

    def Upload(self, request: pb2.ExperienceBatch, context: grpc.ServicerContext) -> empty_pb2.Empty:  # type: ignore[override]
        """Receive a compressed batch and merge it into the buffer."""
        data = gzip.decompress(request.data)
        batch: List[dict] = json.loads(data.decode("utf-8"))
        with self._lock:
            self._experiences.extend(batch)
            self._secure_aggregate()
        return empty_pb2.Empty()

    def Download(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.ExperienceBatch:  # type: ignore[override]
        with self._lock:
            payload = gzip.compress(json.dumps(self._experiences).encode("utf-8"))
        return pb2.ExperienceBatch(data=payload)

    # ------------------------------------------------------------------
    def _secure_aggregate(self) -> None:
        """Replace stored experiences with their noisy average.

        The server keeps only an aggregated representation of uploaded
        experiences.  Individual trades are not retained.  A small amount
        of Gaussian noise is added to each field to make reconstruction of
        individual contributions difficult.
        """

        if not self._experiences:
            return

        states = np.array([e["s"] for e in self._experiences], dtype=np.float32)
        next_states = np.array([e["ns"] for e in self._experiences], dtype=np.float32)
        rewards = np.array([e["r"] for e in self._experiences], dtype=np.float32)
        actions = [int(e["a"]) for e in self._experiences]

        state_mean = states.mean(axis=0)
        next_state_mean = next_states.mean(axis=0)
        reward_mean = float(rewards.mean())
        action_mean = int(round(sum(actions) / len(actions)))

        # add a little noise for privacy
        state_mean += np.random.normal(scale=1e-3, size=state_mean.shape)
        next_state_mean += np.random.normal(scale=1e-3, size=next_state_mean.shape)
        reward_mean += float(np.random.normal(scale=1e-3))

        n = len(self._experiences)
        agg = {
            "s": state_mean.tolist(),
            "ns": next_state_mean.tolist(),
            "r": reward_mean,
            "a": action_mean,
        }
        self._experiences = [agg for _ in range(n)]


def serve(address: str) -> grpc.Server:
    """Start an experience buffer server on ``address``."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_ExperienceBufferServicer_to_server(_BufferServicer(), server)
    server.add_insecure_port(address)
    server.start()
    return server


class FederatedBufferClient:
    """Client helper for uploading and downloading experiences."""

    def __init__(self, address: str) -> None:
        self.address = address

    # ------------------------------------------------------------------
    def upload(self, experiences: Sequence[Experience]) -> None:
        batch = [
            {
                "s": s.tolist(),
                "a": int(a),
                "r": float(r),
                "ns": ns.tolist(),
            }
            for s, a, r, ns in experiences
        ]
        payload = gzip.compress(json.dumps(batch).encode("utf-8"))
        with grpc.insecure_channel(self.address) as channel:
            stub = pb2_grpc.ExperienceBufferStub(channel)
            stub.Upload(pb2.ExperienceBatch(data=payload))

    # ------------------------------------------------------------------
    def download(self) -> List[Experience]:
        with grpc.insecure_channel(self.address) as channel:
            stub = pb2_grpc.ExperienceBufferStub(channel)
            resp = stub.Download(empty_pb2.Empty())
        data = gzip.decompress(resp.data)
        batch: List[dict] = json.loads(data.decode("utf-8"))
        return [
            (
                np.asarray(e["s"], dtype=np.float32),
                int(e["a"]),
                float(e["r"]),
                np.asarray(e["ns"], dtype=np.float32),
            )
            for e in batch
        ]

    # ------------------------------------------------------------------
    def sync(self, experiences: Sequence[Experience]) -> List[Experience]:
        """Upload ``experiences`` and return the aggregated buffer."""
        self.upload(experiences)
        return self.download()


def main() -> None:
    p = argparse.ArgumentParser(description="Federated experience buffer")
    p.add_argument("mode", choices=["server", "upload", "download"])
    p.add_argument("--address", default="127.0.0.1:50051")
    p.add_argument("--file", help="path to JSON file for upload/download")
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()

    logger = setup_logging(__name__)
    set_seed(args.random_seed)

    if args.mode == "server":
        server = serve(args.address)
        logger.info({"event": "server_started", "address": args.address})
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("server shutdown requested")
            server.stop(0)
    elif args.mode == "upload":
        if not args.file:
            raise SystemExit("--file is required for upload")
        with open(args.file) as f:
            batch_json = json.load(f)
        client = FederatedBufferClient(args.address)
        exps = [
            (
                np.asarray(b["s"], dtype=np.float32),
                int(b["a"]),
                float(b["r"]),
                np.asarray(b["ns"], dtype=np.float32),
            )
            for b in batch_json
        ]
        client.upload(exps)
    else:  # download
        client = FederatedBufferClient(args.address)
        exps = client.download()
        if not args.file:
            for e in exps:
                logger.info("%s", e)
        else:
            batch = [
                {
                    "s": s.tolist(),
                    "a": int(a),
                    "r": float(r),
                    "ns": ns.tolist(),
                }
                for s, a, r, ns in exps
            ]
            with open(args.file, "w") as f:
                json.dump(batch, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - manual usage
    main()
