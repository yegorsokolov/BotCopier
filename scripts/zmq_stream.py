#!/usr/bin/env python3
"""Bridge PUSH/PULL input to PUB output using ZeroMQ.

The service listens on a PULL socket for incoming binary messages from the
observer EA and republishes them on a PUB socket so that multiple listeners can
subscribe to the stream.
"""
from __future__ import annotations

import argparse
import logging

import zmq

from otel_logging import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="ZeroMQ stream bridge")
    p.add_argument("--pull", default="tcp://*:5555", help="address to bind the PULL socket")
    p.add_argument("--pub", default="tcp://*:5556", help="address to bind the PUB socket")
    p.add_argument("--log-level", default="INFO", help="logging level")
    args = p.parse_args()

    tracer = setup_logging("zmq_stream", getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger("zmq_stream")

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(args.pull)
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.pub)

    try:
        while True:
            with tracer.start_as_current_span("bridge_message"):
                msg = pull.recv()
                pub.send(msg)
                log.info("forwarded %d bytes", len(msg))
    except KeyboardInterrupt:
        pass
    finally:
        pull.close()
        pub.close()
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
