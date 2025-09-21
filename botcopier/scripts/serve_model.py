import argparse
import base64
import io
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
from botcopier.utils.inference import (
    FeaturePipeline,
    apply_probability_calibration,
    resolve_calibration_metadata,
)

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


def _apply_masked_encoder(raw_features: Sequence[float]) -> list[float]:
    if FEATURE_PIPELINE is None:
        return [float(x) for x in raw_features]
    transformed = FEATURE_PIPELINE.transform_array([float(x) for x in raw_features])
    return transformed.astype(float).tolist()


def _make_logistic_predictor(config: dict[str, object]) -> Callable[[Sequence[float]], float]:
    coeffs = np.asarray(config.get("coefficients", []), dtype=float)
    if coeffs.size == 0:
        raise ValueError("missing logistic coefficients")
    intercept = float(config.get("intercept", 0.0))
    clip_low = np.asarray(config.get("clip_low", []), dtype=float)
    clip_high = np.asarray(config.get("clip_high", []), dtype=float)
    center = np.asarray(config.get("center", []), dtype=float)
    scale = np.asarray(config.get("scale", []), dtype=float)
    calibration_meta, cal_coef, cal_inter = resolve_calibration_metadata(config)

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
        _, calibrated = apply_probability_calibration(
            score,
            _sigmoid(score),
            calibration=calibration_meta,
            legacy_coef=cal_coef,
            legacy_intercept=cal_inter,
        )
        return float(np.asarray(calibrated, dtype=float))

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
    calibration_meta, cal_coef, cal_inter = resolve_calibration_metadata(
        config, MODEL
    )
    _, calibrated = apply_probability_calibration(
        score,
        _sigmoid(score),
        calibration=calibration_meta,
        legacy_coef=cal_coef,
        legacy_intercept=cal_inter,
    )
    return float(np.asarray(calibrated, dtype=float))


def _configure_model(model: dict) -> None:
    """Initialise globals controlling model inference."""

    global MODEL, FEATURE_METADATA, _EXPECTED_COLS, FEATURE_NAMES, INPUT_COLUMNS
    global PT_INFO, PT, PT_IDX
    global OOD_INFO, OOD_MEAN, OOD_COV, OOD_INV, OOD_THRESHOLD
    global LINEAR_CONFIG, ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, THRESHOLD
    global MASKED_ENCODER_INPUTS, FEATURE_PIPELINE

    MODEL = model
    FEATURE_PIPELINE = FeaturePipeline.from_model(model, model_dir=MODEL_DIR)
    FEATURE_METADATA = FEATURE_PIPELINE.feature_metadata
    FEATURE_NAMES = FEATURE_PIPELINE.feature_names
    _EXPECTED_COLS = [
        meta.original_column for meta in FEATURE_METADATA
    ] if FEATURE_METADATA else []
    INPUT_COLUMNS = list(FEATURE_PIPELINE.input_columns or FEATURE_NAMES or FEATURE_COLUMNS)

    PT_INFO = model.get("power_transformer")
    PT = FEATURE_PIPELINE.power_transformer
    PT_IDX = list(FEATURE_PIPELINE.power_indices)

    OOD_INFO = model.get("ood", {})
    OOD_MEAN = np.asarray(OOD_INFO.get("mean", []), dtype=float)
    OOD_COV = np.asarray(OOD_INFO.get("covariance", []), dtype=float)
    OOD_PREC = np.asarray(OOD_INFO.get("precision", []), dtype=float)
    if OOD_PREC.size:
        OOD_INV = OOD_PREC
    elif OOD_COV.size:
        OOD_INV = np.linalg.pinv(OOD_COV)
    else:
        OOD_INV = np.empty((0, 0))
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
        "calibration": model.get("calibration"),
        "calibration_coef": model.get("calibration_coef"),
        "calibration_intercept": model.get("calibration_intercept"),
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
                        "calibration": estimator.get(
                            "calibration", LINEAR_CONFIG.get("calibration")
                        ),
                        "calibration_coef": estimator.get(
                            "calibration_coef", LINEAR_CONFIG.get("calibration_coef")
                        ),
                        "calibration_intercept": estimator.get(
                            "calibration_intercept",
                            LINEAR_CONFIG.get("calibration_intercept"),
                        ),
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

    MASKED_ENCODER_INPUTS = list(FEATURE_PIPELINE.autoencoder_inputs)
    if not INPUT_COLUMNS:
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
FEATURE_PIPELINE: FeaturePipeline | None = None


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

    if FEATURE_PIPELINE is not None:
        transformed = FEATURE_PIPELINE.transform_array(features)
    else:
        transformed = np.asarray(features, dtype=float)
    if transformed.ndim != 1:
        transformed = transformed.ravel()

    if OOD_MEAN.size and OOD_INV.size:
        dist = float(np.sqrt((transformed - OOD_MEAN) @ OOD_INV @ (transformed - OOD_MEAN)))
        if dist > OOD_THRESHOLD:
            OOD_COUNTER.inc()
            return 0.0
    feature_list = transformed.astype(float).tolist()
    if ENSEMBLE_MODELS:
        probabilities = [float(p(feature_list)) for p in ENSEMBLE_MODELS]
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
        prob = _predict_logistic(feature_list, LINEAR_CONFIG)
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
