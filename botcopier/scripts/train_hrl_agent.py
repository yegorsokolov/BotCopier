#!/usr/bin/env python3
"""Train a hierarchical reinforcement learning agent.

This script demonstrates training a simple two level hierarchy where a
meta-controller selects a trading regime and sub-policies decide on
entry/exit actions.  It relies on :mod:`stable_baselines3` when available,
falling back to no-op stubs otherwise so the script remains importable on
systems without the optional dependency.

The resulting hierarchy is saved into ``model.json`` with a ``hierarchy``
field that contains metadata and (when SB3 is available) learned
parameters.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

try:  # pragma: no cover - optional dependency
    import stable_baselines3 as sb3  # type: ignore
    from gym import Env, spaces  # type: ignore

    HAS_SB3 = True
except Exception:  # pragma: no cover - SB3 not installed
    sb3 = None  # type: ignore
    Env = object  # type: ignore
    spaces = None  # type: ignore
    HAS_SB3 = False

try:  # pragma: no cover - used when running from repository root
    from self_play.env import SelfPlayEnv  # type: ignore
except Exception:  # pragma: no cover - import fallback
    try:
        from scripts.self_play_env import SelfPlayEnv  # type: ignore
    except Exception:  # pragma: no cover - environment optional
        SelfPlayEnv = None  # type: ignore


resource = Resource.create(
    {"service.name": os.getenv("OTEL_SERVICE_NAME", "train_hrl_agent")}
)
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
elif os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST") and JaegerExporter:
    provider.add_span_processor(
        BatchSpanProcessor(
            JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )
        )
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
    )
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {"level": record.levelname}
        if isinstance(record.msg, dict):
            log.update(record.msg)
        else:
            log["message"] = record.getMessage()
        if hasattr(record, "trace_id"):
            log["trace_id"] = format_trace_id(record.trace_id)
        if hasattr(record, "span_id"):
            log["span_id"] = format_span_id(record.span_id)
        return json.dumps(log)


logger = logging.getLogger(__name__)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------


class RegimeSelectionEnv(Env):
    """Meta-controller environment.

    The meta-controller chooses among ``n_regimes`` options.  Each step
    simply forwards to the underlying trading environment using a random
    action so that a learning algorithm receives a reward signal.  This is
    intentionally simple – the goal is to provide a minimal example that can
    be expanded for real training pipelines.
    """

    def __init__(self, base_env: Env, n_regimes: int) -> None:
        self.base_env = base_env
        self.n_regimes = n_regimes
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Discrete(n_regimes)

    def reset(self) -> Any:
        return self.base_env.reset()

    def step(self, action: int):
        # Select a random action in the underlying environment to generate a
        # reward signal for the chosen regime.
        sub_action = self.base_env.action_space.sample()
        obs, reward, done, info = self.base_env.step(sub_action)
        info = dict(info)
        info["regime"] = int(action)
        return obs, reward, done, info


class RegimeEnv(Env):
    """Wrapper used to train a sub-policy for a specific regime."""

    def __init__(self, base_env: Env, regime: int) -> None:
        self.base_env = base_env
        self.regime = regime
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(self) -> Any:  # pragma: no cover - simple forwarding
        return self.base_env.reset()

    def step(self, action):  # pragma: no cover - simple forwarding
        return self.base_env.step(action)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _to_jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SB3 parameters (numpy arrays) to plain lists for JSON."""

    simple: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            simple[k] = v.tolist()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - non-array values
            simple[k] = v
    return simple


def train(
    output: Path, meta_steps: int = 1000, sub_steps: int = 1000, regimes: int = 2
) -> None:
    """Train the hierarchical agent and write ``model.json``."""
    if not SelfPlayEnv:
        raise RuntimeError("SelfPlayEnv is not available")
    with tracer.start_as_current_span("train_hrl_agent_train"):
        logger.info("training hierarchical agent")
        env = SelfPlayEnv()

        hierarchy: Dict[str, Any] = {"type": "options", "sub_policies": {}}

        if HAS_SB3:
            # Train meta-controller
            meta_env = RegimeSelectionEnv(env, regimes)
            meta_model = sb3.PPO("MlpPolicy", meta_env, verbose=0)
            meta_model.learn(total_timesteps=meta_steps)
            hierarchy["meta_controller"] = {
                "algorithm": "PPO",
                "parameters": _to_jsonable(meta_model.get_parameters()),
            }

            # Train sub-policies for each regime
            for r in range(regimes):
                sub_env = RegimeEnv(env, r)
                sub_model = sb3.PPO("MlpPolicy", sub_env, verbose=0)
                sub_model.learn(total_timesteps=sub_steps)
                hierarchy["sub_policies"][str(r)] = {
                    "algorithm": "PPO",
                    "parameters": _to_jsonable(sub_model.get_parameters()),
                }
        else:  # pragma: no cover - executed when SB3 missing
            logging.warning(
                "stable-baselines3 not installed; producing empty hierarchy"
            )
            hierarchy["meta_controller"] = {}

        data = {"hierarchy": hierarchy}
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("model saved", extra={"output": str(output)})


def main() -> None:  # pragma: no cover - thin CLI wrapper
    span = tracer.start_span("train_hrl_agent")
    ctx = span.get_span_context()
    logger.info("start", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("model.json"))
    p.add_argument("--meta-steps", type=int, default=1000)
    p.add_argument("--sub-steps", type=int, default=1000)
    p.add_argument("--regimes", type=int, default=2)
    args = p.parse_args()
    train(args.output, args.meta_steps, args.sub_steps, args.regimes)
    logger.info("finished", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})
    span.end()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
