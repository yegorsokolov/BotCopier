import argparse
import json
import math
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import PowerTransformer

from botcopier.metrics import (
    ERROR_COUNTER,
    TRADE_COUNTER,
    observe_latency,
    start_metrics_server,
)
from botcopier.models.schema import FeatureMetadata

try:  # optional feast dependency
    from feast import FeatureStore  # type: ignore

    from botcopier.feature_store.feast_repo.feature_views import FEATURE_COLUMNS

    _HAS_FEAST = True
except Exception:  # pragma: no cover - optional
    FeatureStore = None  # type: ignore
    FEATURE_COLUMNS = []  # type: ignore
    _HAS_FEAST = False

# Load model.json at startup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.json"
try:
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        MODEL = json.load(f)
except FileNotFoundError:
    MODEL = {"feature_names": [], "entry_coefficients": [], "entry_intercept": 0.0}

FEATURE_METADATA = [FeatureMetadata(**m) for m in MODEL.get("feature_metadata", [])]
_EXPECTED_COLS = [fm.original_column for fm in FEATURE_METADATA]
FEATURE_NAMES = MODEL.get("feature_names", _EXPECTED_COLS or FEATURE_COLUMNS)
if FEATURE_METADATA and len(FEATURE_NAMES) != len(FEATURE_METADATA):
    raise ValueError("feature_names and feature_metadata mismatch")

PT_INFO = MODEL.get("power_transformer")
PT: PowerTransformer | None = None
PT_IDX: List[int] = []
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
    """Simple linear model with sigmoid activation."""

    coeffs = MODEL.get("entry_coefficients", [])
    intercept = MODEL.get("entry_intercept", 0.0)
    expected = len(FEATURE_METADATA) or len(FEATURE_NAMES)
    if len(coeffs) != expected:
        raise ValueError("model definition inconsistent")
    if len(features) != expected:
        raise ValueError("feature length mismatch")
    score = sum(c * f for c, f in zip(coeffs, features)) + intercept
    return 1.0 / (1.0 + math.exp(-score))


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """Return predictions for a batch of symbols."""
    if not _HAS_FEAST or STORE is None:
        raise HTTPException(status_code=500, detail="feature store unavailable")
    with observe_latency("predict"):
        feature_cols = _EXPECTED_COLS or FEATURE_COLUMNS
        feature_refs = [f"trade_features:{f}" for f in feature_cols]
        entity_rows = [{"symbol": s} for s in req.symbols]
        feat_dict = STORE.get_online_features(
            features=feature_refs, entity_rows=entity_rows
        ).to_dict()
        results: List[float] = []
        for i in range(len(req.symbols)):
            try:
                features = [feat_dict[f][i] for f in feature_cols]
            except KeyError as exc:
                ERROR_COUNTER.labels(type="predict").inc()
                raise HTTPException(
                    status_code=400,
                    detail=f"missing feature '{exc.args[0]}'",
                ) from exc
            try:
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


def main() -> None:
    """Run a small FastAPI service that serves the trained model."""

    parser = argparse.ArgumentParser(description="Serve the distilled model")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--metrics-port", type=int, default=8004, help="Prometheus metrics port"
    )
    args = parser.parse_args()
    start_metrics_server(args.metrics_port)
    uvicorn.run("botcopier.scripts.serve_model:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
