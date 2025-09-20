#!/usr/bin/env python3
"""gRPC service exposing a simple model prediction RPC."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import math
import pickle
import signal
from contextlib import suppress
from pathlib import Path
from typing import Callable, List, Sequence

import grpc.aio as grpc

import numpy as np

try:  # optional dependency used for masked encoder checkpoints
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _HAS_TORCH = False

# Make generated proto modules importable
PROTO_DIR = Path(__file__).resolve().parent.parent.parent / "proto"
import sys
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore

from botcopier.exceptions import ModelError, ServiceError

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.json"

MODEL: dict = {"entry_coefficients": [], "entry_intercept": 0.0}
COEFFS: List[float] = []
INTERCEPT: float = 0.0
LINEAR_CONFIG: dict[str, object] = {}
ENSEMBLE_MODELS: list[Callable[[Sequence[float]], float]] = []
ENSEMBLE_WEIGHTS: Sequence[float] | None = None
THRESHOLD: float = 0.5
MODEL_DIR: Path | None = DEFAULT_MODEL_PATH.parent
MASKED_ENCODER_WEIGHTS = np.empty((0, 0))
MASKED_ENCODER_BIAS = np.empty(0)
MASKED_ENCODER_INPUT_DIM = 0

logger = logging.getLogger(__name__)


def _load_model(path: Path) -> dict:
    """Load a JSON model specification from ``path``."""

    try:
        with path.open("r", encoding="utf-8") as fh:
            model = json.load(fh)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModelError(f"Model file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelError(f"Model file is not valid JSON: {path}") from exc
    except OSError as exc:  # pragma: no cover - defensive guard
        raise ModelError(f"Unable to read model file: {path}") from exc

    if not isinstance(model, dict):
        raise ModelError(f"Model payload must be a JSON object: {path}")
    return model


def _sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


def _resolve_threshold(values: Sequence[object]) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.5


def _load_masked_encoder_from_file(weights_file: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not weights_file:
        return None, None
    path = Path(str(weights_file))
    if not path.is_absolute():
        if MODEL_DIR is not None:
            path = (MODEL_DIR / path).resolve()
        else:
            path = path.resolve()
    if not path.exists():
        logger.warning("masked encoder weights file not found: %s", path)
        return None, None
    if not _HAS_TORCH:
        logger.warning(
            "masked encoder weights file present at %s but torch is unavailable",
            path,
        )
        return None, None
    try:
        assert torch is not None  # for type checkers
        state = torch.load(path, map_location="cpu")
    except Exception:  # pragma: no cover - optional dependency
        logger.exception("failed to load masked encoder weights from %s", path)
        return None, None
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else {}
    weight_tensor = state_dict.get("weight")
    bias_tensor = state_dict.get("bias")
    if isinstance(weight_tensor, torch.Tensor):
        weight_arr = weight_tensor.detach().cpu().numpy()
    else:
        weight_arr = (
            np.asarray(weight_tensor, dtype=float)
            if weight_tensor is not None
            else None
        )
    if isinstance(bias_tensor, torch.Tensor):
        bias_arr = bias_tensor.detach().cpu().numpy()
    else:
        bias_arr = np.asarray(bias_tensor, dtype=float) if bias_tensor is not None else None
    return weight_arr, bias_arr


def _initialise_masked_encoder(meta: dict[str, object] | None) -> None:
    """Configure globals describing the optional masked encoder."""

    global MASKED_ENCODER_WEIGHTS, MASKED_ENCODER_BIAS, MASKED_ENCODER_INPUT_DIM

    MASKED_ENCODER_WEIGHTS = np.empty((0, 0))
    MASKED_ENCODER_BIAS = np.empty(0)
    MASKED_ENCODER_INPUT_DIM = 0

    if not isinstance(meta, dict):
        return

    inputs = meta.get("input_features") or meta.get("original_features")
    if isinstance(inputs, Sequence):
        MASKED_ENCODER_INPUT_DIM = len(list(inputs))
    elif isinstance(inputs, str):
        MASKED_ENCODER_INPUT_DIM = 1

    weights = meta.get("weights")
    bias = meta.get("bias")
    weight_arr: np.ndarray | None
    bias_arr: np.ndarray | None
    if weights is not None:
        weight_arr = np.asarray(weights, dtype=float)
        bias_arr = np.asarray(bias, dtype=float) if bias is not None else None
    else:
        weight_arr, bias_arr = _load_masked_encoder_from_file(meta.get("weights_file"))
    if weight_arr is None:
        if MASKED_ENCODER_INPUT_DIM:
            logger.warning("masked encoder metadata present but weights unavailable")
        return

    MASKED_ENCODER_WEIGHTS = weight_arr.astype(float)
    if bias_arr is not None:
        MASKED_ENCODER_BIAS = bias_arr.astype(float)
    if not MASKED_ENCODER_INPUT_DIM and MASKED_ENCODER_WEIGHTS.size:
        MASKED_ENCODER_INPUT_DIM = MASKED_ENCODER_WEIGHTS.shape[1]


def _apply_masked_encoder(features: Sequence[float]) -> list[float]:
    if MASKED_ENCODER_WEIGHTS.size == 0:
        return [float(x) for x in features]
    arr = np.asarray(features, dtype=float)
    expected = MASKED_ENCODER_WEIGHTS.shape[1]
    if MASKED_ENCODER_INPUT_DIM and arr.size != MASKED_ENCODER_INPUT_DIM:
        raise ServiceError(
            f"masked encoder expected {MASKED_ENCODER_INPUT_DIM} inputs, received {arr.size}"
        )
    if arr.size != expected:
        raise ServiceError(
            f"masked encoder weight matrix expects {expected} inputs, received {arr.size}"
        )
    encoded = arr @ MASKED_ENCODER_WEIGHTS.T
    if MASKED_ENCODER_BIAS.size == encoded.shape[0]:
        encoded = encoded + MASKED_ENCODER_BIAS
    return encoded.astype(float).tolist()


def _make_logistic_predictor(config: dict[str, object]) -> Callable[[Sequence[float]], float]:
    coeffs = np.asarray(config.get("coefficients", []), dtype=float)
    if coeffs.size == 0:
        raise ValueError("missing logistic coefficients")
    intercept = float(config.get("intercept", 0.0))
    clip_low = np.asarray(config.get("clip_low", []), dtype=float)
    clip_high = np.asarray(config.get("clip_high", []), dtype=float)
    center = np.asarray(config.get("center", []), dtype=float)
    scale = np.asarray(config.get("scale", []), dtype=float)

    def _predict(features: Sequence[float]) -> float:
        arr = np.asarray(features, dtype=float)
        if arr.size != coeffs.size:
            raise ServiceError("feature length mismatch")
        if clip_low.size == arr.size and clip_high.size == arr.size:
            arr = np.clip(arr, clip_low, clip_high)
        if center.size == arr.size and scale.size == arr.size:
            safe_scale = np.where(scale == 0, 1.0, scale)
            arr = (arr - center) / safe_scale
        score = float(np.dot(coeffs, arr) + intercept)
        return float(_sigmoid(score))

    return _predict


def _make_gradient_boosting_predictor(config: dict[str, object]) -> Callable[[Sequence[float]], float]:
    payload = config.get("model")
    if not payload:
        raise ValueError("missing gradient boosting payload")
    model = pickle.loads(base64.b64decode(payload))

    def _predict(features: Sequence[float]) -> float:
        arr = np.asarray(features, dtype=float).reshape(1, -1)
        prob = model.predict_proba(arr)[0, 1]
        return float(prob)

    return _predict


def _make_xgboost_predictor(config: dict[str, object]) -> Callable[[Sequence[float]], float]:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("xgboost is required to load this estimator") from exc

    booster_payload = config.get("booster")
    if not booster_payload:
        raise ValueError("missing xgboost booster payload")
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(booster_payload)))

    def _predict(features: Sequence[float]) -> float:
        arr = np.asarray(features, dtype=float).reshape(1, -1)
        dmatrix = xgb.DMatrix(arr)
        prob = booster.predict(dmatrix)
        return float(prob[0])

    return _predict


def _make_catboost_predictor(config: dict[str, object]) -> Callable[[Sequence[float]], float]:
    try:
        import catboost as cb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("catboost is required to load this estimator") from exc

    payload = config.get("cb_model") or config.get("model")
    if not payload:
        raise ValueError("missing catboost model payload")
    buffer = io.BytesIO(base64.b64decode(payload))
    model = cb.CatBoostClassifier()
    model.load_model(stream=buffer)

    def _predict(features: Sequence[float]) -> float:
        arr = np.asarray(features, dtype=float).reshape(1, -1)
        prob = model.predict_proba(arr)[0, 1]
        return float(prob)

    return _predict


def _predict_logistic(features: Sequence[float]) -> float:
    if not LINEAR_CONFIG.get("coefficients"):
        if not COEFFS:
            raise ServiceError("model coefficients unavailable")
        coeffs = np.asarray(COEFFS, dtype=float)
        intercept = float(INTERCEPT)
        clip_low = np.asarray([], dtype=float)
        clip_high = np.asarray([], dtype=float)
        center = np.asarray([], dtype=float)
        scale = np.asarray([], dtype=float)
    else:
        coeffs = np.asarray(LINEAR_CONFIG.get("coefficients", []), dtype=float)
        intercept = float(LINEAR_CONFIG.get("intercept", INTERCEPT))
        clip_low = np.asarray(LINEAR_CONFIG.get("clip_low", []), dtype=float)
        clip_high = np.asarray(LINEAR_CONFIG.get("clip_high", []), dtype=float)
        center = np.asarray(LINEAR_CONFIG.get("center", []), dtype=float)
        scale = np.asarray(LINEAR_CONFIG.get("scale", []), dtype=float)

    arr = np.asarray(features, dtype=float)
    if arr.size != coeffs.size:
        raise ServiceError("feature length mismatch")
    if clip_low.size == arr.size and clip_high.size == arr.size:
        arr = np.clip(arr, clip_low, clip_high)
    if center.size == arr.size and scale.size == arr.size:
        safe_scale = np.where(scale == 0, 1.0, scale)
        arr = (arr - center) / safe_scale
    score = float(np.dot(coeffs, arr) + intercept)
    return float(_sigmoid(score))


def _configure_runtime(model: dict) -> None:
    """Populate globals used during inference from ``model``."""

    global MODEL, COEFFS, INTERCEPT, LINEAR_CONFIG, ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, THRESHOLD
    MODEL = model
    coeffs = model.get("entry_coefficients") or model.get("coefficients") or []
    COEFFS = [float(x) for x in coeffs]
    INTERCEPT = float(model.get("entry_intercept", model.get("intercept", 0.0)))
    LINEAR_CONFIG = {
        "coefficients": COEFFS,
        "intercept": INTERCEPT,
        "clip_low": model.get("clip_low", []),
        "clip_high": model.get("clip_high", []),
        "center": model.get("feature_mean", []),
        "scale": model.get("feature_std", []),
    }
    ENSEMBLE_MODELS = []
    ENSEMBLE_WEIGHTS = None
    threshold_candidates: list[object] = []
    ensemble_cfg = model.get("ensemble")
    if isinstance(ensemble_cfg, dict):
        weights = ensemble_cfg.get("weights")
        if weights is not None:
            ENSEMBLE_WEIGHTS = [float(w) for w in weights]
        threshold_candidates.append(ensemble_cfg.get("threshold"))
        for estimator in ensemble_cfg.get("estimators", []):
            est_type = estimator.get("type")
            try:
                if est_type == "logistic":
                    ENSEMBLE_MODELS.append(_make_logistic_predictor(estimator))
                    coeffs_local = [float(x) for x in estimator.get("coefficients", [])]
                    COEFFS = coeffs_local or COEFFS
                    INTERCEPT = float(estimator.get("intercept", INTERCEPT))
                    LINEAR_CONFIG = {
                        "coefficients": coeffs_local or COEFFS,
                        "intercept": INTERCEPT,
                        "clip_low": estimator.get("clip_low", []),
                        "clip_high": estimator.get("clip_high", []),
                        "center": estimator.get("center", []),
                        "scale": estimator.get("scale", []),
                    }
                elif est_type == "gradient_boosting":
                    ENSEMBLE_MODELS.append(_make_gradient_boosting_predictor(estimator))
                elif est_type == "xgboost":
                    ENSEMBLE_MODELS.append(_make_xgboost_predictor(estimator))
                else:
                    logger.warning("Unsupported estimator type '%s'", est_type)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to initialise estimator %s", estimator.get("name", est_type)
                )
    standalone_estimators: list[tuple[str, Callable[[dict[str, object]], Callable[[Sequence[float]], float]], dict[str, object]]] = []
    gb_model = model.get("gb_model")
    if gb_model:
        standalone_estimators.append(
            ("gradient_boosting", _make_gradient_boosting_predictor, {"model": gb_model})
        )
    booster_payload = model.get("booster")
    if booster_payload:
        standalone_estimators.append(
            ("xgboost", _make_xgboost_predictor, {"booster": booster_payload})
        )
    cb_model = model.get("cb_model")
    if cb_model:
        standalone_estimators.append(
            ("catboost", _make_catboost_predictor, {"cb_model": cb_model})
        )
    for name, factory, cfg in standalone_estimators:
        try:
            ENSEMBLE_MODELS.append(factory(cfg))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialise standalone %s predictor", name)

    threshold_candidates.extend([model.get("decision_threshold"), model.get("threshold")])
    THRESHOLD = _resolve_threshold(threshold_candidates)
    _initialise_masked_encoder(model.get("masked_encoder"))
    logger.info(
        "model_loaded",
        extra={
            "context": {
                "coefficients": len(COEFFS),
                "intercept": INTERCEPT,
                "threshold": THRESHOLD,
                "ensemble_estimators": len(ENSEMBLE_MODELS),
            }
        },
    )

def _reload_model(path: Path) -> None:
    """Refresh global model parameters from ``path``."""

    global MODEL_DIR

    model = _load_model(path)
    MODEL_DIR = path.parent.resolve()
    _configure_runtime(model)
    logger.info("model_source", extra={"context": {"path": str(path)}})


def _predict_one(features: List[float]) -> float:
    """Return ensemble probability applying the configured threshold."""

    features = _apply_masked_encoder(features)
    if ENSEMBLE_MODELS:
        probabilities: list[float] = []
        for predictor in ENSEMBLE_MODELS:
            probabilities.append(float(predictor(features)))
        if not probabilities:
            raise ServiceError("no ensemble estimators available")
        if ENSEMBLE_WEIGHTS is not None and len(ENSEMBLE_WEIGHTS) == len(probabilities):
            weight_sum = float(sum(ENSEMBLE_WEIGHTS))
            if weight_sum > 0:
                prob = float(
                    sum(w * p for w, p in zip(ENSEMBLE_WEIGHTS, probabilities)) / weight_sum
                )
            else:
                prob = float(sum(probabilities) / len(probabilities))
        else:
            prob = float(sum(probabilities) / len(probabilities))
    else:
        prob = _predict_logistic(features)
    return prob if prob >= THRESHOLD else 0.0


def _build_server(host: str, port: int) -> grpc.Server:
    server = grpc.server()
    predict_pb2_grpc.add_PredictServiceServicer_to_server(_PredictService(), server)
    server.add_insecure_port(f"{host}:{port}")
    return server


class _ServerManager:
    """Async context manager ensuring the gRPC server is stopped."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self.server: grpc.Server | None = None
        self.stopped = False

    async def __aenter__(self) -> grpc.Server:
        server = _build_server(self._host, self._port)
        await server.start()
        logger.info(
            "server_started",
            extra={"context": {"host": self._host, "port": self._port}},
        )
        self.server = server
        return server

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.server and not self.stopped:
            await self.server.stop(None)
        logger.info(
            "server_stopped",
            extra={"context": {"host": self._host, "port": self._port}},
        )


class _PredictService(predict_pb2_grpc.PredictServiceServicer):
    async def Predict(self, request, context):  # noqa: N802 gRPC naming
        try:
            pred = _predict_one(list(request.features))
            return predict_pb2.PredictResponse(prediction=pred)
        except ServiceError as exc:
            logger.exception(
                "prediction_failed",
                extra={
                    "context": {
                        "feature_count": len(request.features),
                        "error": str(exc),
                    }
                },
            )
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "prediction_internal_error",
                extra={
                    "context": {
                        "feature_count": len(request.features),
                        "error": str(exc),
                    }
                },
            )
            await context.abort(grpc.StatusCode.INTERNAL, "internal error")


async def create_server(host: str, port: int) -> grpc.Server:
    server = _build_server(host, port)
    await server.start()
    logger.info(
        "server_started",
        extra={"context": {"host": host, "port": port}},
    )
    return server


async def serve(
    host: str,
    port: int,
    *,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    stop_event = shutdown_event or asyncio.Event()
    registered: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
            registered.append(sig)
        except NotImplementedError:  # pragma: no cover - platform specific
            continue

    manager = _ServerManager(host, port)
    try:
        async with manager as server:
            wait_task = asyncio.create_task(server.wait_for_termination())
            stop_task = asyncio.create_task(stop_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done and not wait_task.done():
                    logger.info(
                        "shutdown_requested",
                        extra={"context": {"host": host, "port": port}},
                    )
                    await server.stop(None)
                    manager.stopped = True
                await wait_task
                manager.stopped = True
            finally:
                for task in (wait_task, stop_task):
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
    finally:
        for sig in registered:
            loop.remove_signal_handler(sig)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="gRPC predict service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    try:
        _reload_model(args.model)
    except ModelError:
        logger.exception(
            "model_load_failed", extra={"context": {"path": str(args.model)}}
        )
        raise SystemExit(1) from None

    await serve(args.host, args.port)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
