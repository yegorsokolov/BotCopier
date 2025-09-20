import argparse
import base64
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
try:  # optional server dependency
    import uvicorn
except ImportError:  # pragma: no cover - optional dependency
    uvicorn = None  # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pandera.errors import SchemaErrors
from pydantic import BaseModel
from sklearn.preprocessing import PowerTransformer

from botcopier.metrics import (
    ERROR_COUNTER,
    TRADE_COUNTER,
    OOD_COUNTER,
    latest_metrics,
    observe_latency,
    start_metrics_server,
)
from botcopier.data.feature_schema import FeatureSchema
from botcopier.models.schema import FeatureMetadata

try:  # optional feast dependency
    from feast import FeatureStore  # type: ignore

    from botcopier.feature_store.feast_repo.feature_views import FEATURE_COLUMNS

    _HAS_FEAST = True
except Exception:  # pragma: no cover - optional
    FeatureStore = None  # type: ignore
    FEATURE_COLUMNS = []  # type: ignore
    _HAS_FEAST = False


logger = logging.getLogger(__name__)


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
        path = (MODEL_DIR / path).resolve()
    if not path.exists():
        logger.warning("Masked encoder weights file not found at %s", path)
        return None, None
    try:
        import torch  # type: ignore

        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state)
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
    except Exception:  # pragma: no cover - optional torch dependency
        logger.exception("Failed to load masked encoder weights from %s", path)
        return None, None


def _initialise_masked_encoder(meta: dict[str, object] | None) -> None:
    """Populate globals describing the optional masked encoder."""

    global MASKED_ENCODER_WEIGHTS, MASKED_ENCODER_BIAS, MASKED_ENCODER_INPUTS

    MASKED_ENCODER_WEIGHTS = np.empty((0, 0))
    MASKED_ENCODER_BIAS = np.empty(0)
    MASKED_ENCODER_INPUTS = []

    if not isinstance(meta, dict):
        return

    inputs = meta.get("input_features") or meta.get("original_features")
    if isinstance(inputs, Sequence):
        MASKED_ENCODER_INPUTS = [str(col) for col in inputs]

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
        if MASKED_ENCODER_INPUTS:
            logger.warning("Masked encoder metadata present but weights unavailable")
        return
    MASKED_ENCODER_WEIGHTS = weight_arr.astype(float)
    if bias_arr is not None:
        MASKED_ENCODER_BIAS = bias_arr.astype(float)


def _apply_masked_encoder(raw_features: Sequence[float]) -> list[float]:
    if MASKED_ENCODER_WEIGHTS.size == 0:
        return [float(x) for x in raw_features]
    arr = np.asarray(raw_features, dtype=float)
    expected = MASKED_ENCODER_WEIGHTS.shape[1]
    if arr.size != expected:
        raise ValueError(
            f"masked encoder expected {expected} features, received {arr.size}"
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
            raise ValueError("feature length mismatch")
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


def _predict_logistic(features: Sequence[float], config: dict[str, object]) -> float:
    coeffs = np.asarray(config.get("coefficients", []), dtype=float)
    intercept = float(config.get("intercept", 0.0))
    if coeffs.size == 0:
        raise ValueError("model coefficients unavailable")
    arr = np.asarray(features, dtype=float)
    if arr.size != coeffs.size:
        raise ValueError("feature length mismatch")
    clip_low = np.asarray(config.get("clip_low", []), dtype=float)
    clip_high = np.asarray(config.get("clip_high", []), dtype=float)
    if clip_low.size == arr.size and clip_high.size == arr.size:
        arr = np.clip(arr, clip_low, clip_high)
    center = np.asarray(config.get("center", []), dtype=float)
    scale = np.asarray(config.get("scale", []), dtype=float)
    if center.size == arr.size and scale.size == arr.size:
        safe_scale = np.where(scale == 0, 1.0, scale)
        arr = (arr - center) / safe_scale
    score = float(np.dot(coeffs, arr) + intercept)
    return float(_sigmoid(score))


def _configure_model(model: dict) -> None:
    """Initialise globals controlling model inference."""

    global MODEL, FEATURE_METADATA, _EXPECTED_COLS, FEATURE_NAMES, INPUT_COLUMNS
    global PT_INFO, PT, PT_IDX
    global OOD_INFO, OOD_MEAN, OOD_COV, OOD_INV, OOD_THRESHOLD
    global LINEAR_CONFIG, ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, THRESHOLD
    global MASKED_ENCODER_INPUTS

    MODEL = model
    FEATURE_METADATA = [FeatureMetadata(**m) for m in model.get("feature_metadata", [])]
    _EXPECTED_COLS = [fm.original_column for fm in FEATURE_METADATA]
    FEATURE_NAMES = model.get("feature_names", _EXPECTED_COLS or FEATURE_COLUMNS)
    if FEATURE_METADATA and len(FEATURE_NAMES) != len(FEATURE_METADATA):
        raise ValueError("feature_names and feature_metadata mismatch")

    INPUT_COLUMNS = list(_EXPECTED_COLS or FEATURE_NAMES or FEATURE_COLUMNS)

    PT_INFO = model.get("power_transformer")
    PT = None
    PT_IDX = []
    if PT_INFO:
        PT = PowerTransformer(method="yeo-johnson")
        PT.lambdas_ = np.asarray(PT_INFO.get("lambdas", []), dtype=float)
        PT.n_features_in_ = PT.lambdas_.shape[0]
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(PT_INFO.get("mean", []), dtype=float)
        scaler.scale_ = np.asarray(PT_INFO.get("scale", []), dtype=float)
        PT._scaler = scaler
        PT_IDX = [
            FEATURE_NAMES.index(f)
            for f in PT_INFO.get("features", [])
            if f in FEATURE_NAMES
        ]

    OOD_INFO = model.get("ood", {})
    OOD_MEAN = np.asarray(OOD_INFO.get("mean", []), dtype=float)
    OOD_COV = np.asarray(OOD_INFO.get("covariance", []), dtype=float)
    OOD_INV = np.linalg.pinv(OOD_COV) if OOD_COV.size else np.empty((0, 0))
    OOD_THRESHOLD = float(OOD_INFO.get("threshold", float("inf")))

    coeffs = model.get("entry_coefficients") or model.get("coefficients") or []
    intercept = float(model.get("entry_intercept", model.get("intercept", 0.0)))
    LINEAR_CONFIG = {
        "coefficients": [float(x) for x in coeffs],
        "intercept": intercept,
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
                    LINEAR_CONFIG = {
                        "coefficients": [
                            float(x)
                            for x in estimator.get(
                                "coefficients", LINEAR_CONFIG.get("coefficients", [])
                            )
                        ],
                        "intercept": float(
                            estimator.get("intercept", LINEAR_CONFIG.get("intercept", 0.0))
                        ),
                        "clip_low": estimator.get(
                            "clip_low", LINEAR_CONFIG.get("clip_low", [])
                        ),
                        "clip_high": estimator.get(
                            "clip_high", LINEAR_CONFIG.get("clip_high", [])
                        ),
                        "center": estimator.get("center", LINEAR_CONFIG.get("center", [])),
                        "scale": estimator.get("scale", LINEAR_CONFIG.get("scale", [])),
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
    threshold_candidates.extend([model.get("decision_threshold"), model.get("threshold")])
    THRESHOLD = _resolve_threshold(threshold_candidates)

    encoder_meta = model.get("masked_encoder")
    _initialise_masked_encoder(encoder_meta if isinstance(encoder_meta, dict) else None)
    if MASKED_ENCODER_INPUTS:
        INPUT_COLUMNS = list(MASKED_ENCODER_INPUTS)
        _EXPECTED_COLS = list(MASKED_ENCODER_INPUTS)
    elif not INPUT_COLUMNS:
        INPUT_COLUMNS = list(FEATURE_COLUMNS)


MODEL: dict[str, object] = {"feature_names": [], "entry_coefficients": [], "entry_intercept": 0.0}
FEATURE_METADATA: list[FeatureMetadata] = []
_EXPECTED_COLS: list[str] = []
FEATURE_NAMES: list[str] = []
INPUT_COLUMNS: list[str] = []
PT_INFO: dict[str, object] | None = None
PT: PowerTransformer | None = None
PT_IDX: List[int] = []
OOD_INFO: dict[str, object] = {}
OOD_MEAN = np.empty(0)
OOD_COV = np.empty((0, 0))
OOD_INV = np.empty((0, 0))
OOD_THRESHOLD: float = float("inf")
LINEAR_CONFIG: dict[str, object] = {}
ENSEMBLE_MODELS: list[Callable[[Sequence[float]], float]] = []
ENSEMBLE_WEIGHTS: Sequence[float] | None = None
THRESHOLD: float = 0.5
MASKED_ENCODER_INPUTS: list[str] = []
MASKED_ENCODER_WEIGHTS = np.empty((0, 0))
MASKED_ENCODER_BIAS = np.empty(0)


# Load model.json at startup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.json"
MODEL_DIR = MODEL_PATH.parent
try:
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        MODEL = json.load(f)
except FileNotFoundError:
    MODEL = {"feature_names": [], "entry_coefficients": [], "entry_intercept": 0.0}

_configure_model(MODEL)

if _HAS_FEAST:
    FS_REPO = BASE_DIR / "feature_store" / "feast_repo"
    STORE = FeatureStore(repo_path=str(FS_REPO))
else:
    STORE = None

app = FastAPI(title="BotCopier Model Server")


class PredictionRequest(BaseModel):
    """Request payload containing entity identifiers."""

    symbols: List[str]


class PredictionResponse(BaseModel):
    """Model predictions for each input batch."""

    predictions: List[float]


def _predict_one(features: List[float]) -> float:
    """Compute an ensemble prediction applying the configured threshold."""

    expected = len(FEATURE_METADATA) or len(FEATURE_NAMES)
    if expected and len(features) != expected:
        raise ValueError("feature length mismatch")
    if OOD_MEAN.size and OOD_INV.size:
        arr = np.asarray(features, dtype=float)
        dist = float(np.sqrt((arr - OOD_MEAN) @ OOD_INV @ (arr - OOD_MEAN)))
        if dist > OOD_THRESHOLD:
            OOD_COUNTER.inc()
            return 0.0
    if ENSEMBLE_MODELS:
        probabilities = [float(p(features)) for p in ENSEMBLE_MODELS]
        if not probabilities:
            raise ValueError("no ensemble estimators available")
        if ENSEMBLE_WEIGHTS is not None and len(ENSEMBLE_WEIGHTS) == len(probabilities):
            weight_sum = float(sum(ENSEMBLE_WEIGHTS))
            if weight_sum > 0:
                prob = float(
                    sum(w * p for w, p in zip(ENSEMBLE_WEIGHTS, probabilities))
                    / weight_sum
                )
            else:
                prob = float(sum(probabilities) / len(probabilities))
        else:
            prob = float(sum(probabilities) / len(probabilities))
    else:
        prob = _predict_logistic(features, LINEAR_CONFIG)
    return prob if prob >= THRESHOLD else 0.0


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """Return predictions for a batch of symbols."""
    if not _HAS_FEAST or STORE is None:
        raise HTTPException(status_code=500, detail="feature store unavailable")
    with observe_latency("predict"):
        feature_cols = INPUT_COLUMNS or FEATURE_COLUMNS
        feature_refs = [f"trade_features:{f}" for f in feature_cols]
        entity_rows = [{"symbol": s} for s in req.symbols]
        feat_dict = STORE.get_online_features(
            features=feature_refs, entity_rows=entity_rows
        ).to_dict()
        try:
            frame = pd.DataFrame({col: feat_dict[col] for col in feature_cols})
        except KeyError as exc:
            ERROR_COUNTER.labels(type="predict").inc()
            raise HTTPException(
                status_code=400,
                detail=f"missing feature '{exc.args[0]}'",
            ) from exc
        try:
            if not frame.empty:
                FeatureSchema.validate(frame, lazy=True)
        except SchemaErrors as exc:
            ERROR_COUNTER.labels(type="predict").inc()
            raise HTTPException(
                status_code=400,
                detail="feature schema validation failed",
            ) from exc
        results: List[float] = []
        for i in range(len(req.symbols)):
            features = [float(frame.iloc[i][col]) for col in feature_cols]
            try:
                features = _apply_masked_encoder(features)
                if PT is not None and PT_IDX:
                    arr = np.asarray([features[j] for j in PT_IDX], dtype=float).reshape(1, -1)
                    transformed = PT.transform(arr).ravel().tolist()
                    for j, val in zip(PT_IDX, transformed):
                        features[j] = float(val)
                results.append(_predict_one(features))
            except ValueError as exc:
                ERROR_COUNTER.labels(type="predict").inc()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        TRADE_COUNTER.inc(len(req.symbols))
    return PredictionResponse(predictions=results)


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for the model server."""

    payload, content_type = latest_metrics()
    return Response(content=payload, media_type=content_type)


def main() -> None:
    """Run a small FastAPI service that serves the trained model."""

    parser = argparse.ArgumentParser(description="Serve the distilled model")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--metrics-port", type=int, default=8004, help="Prometheus metrics port"
    )
    args = parser.parse_args()
    if uvicorn is None:  # pragma: no cover - optional dependency
        raise RuntimeError("uvicorn is required to run the model server")
    start_metrics_server(args.metrics_port)
    uvicorn.run("botcopier.scripts.serve_model:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
