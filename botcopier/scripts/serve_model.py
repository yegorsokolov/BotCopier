import argparse
import json
import math
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model.json at startup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.json"
try:
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        MODEL = json.load(f)
except FileNotFoundError:
    MODEL = {"feature_names": [], "entry_coefficients": [], "entry_intercept": 0.0}
FEATURE_NAMES = MODEL.get("feature_names", [])

app = FastAPI(title="BotCopier Model Server")


class PredictionRequest(BaseModel):
    """Request payload containing feature batches."""

    instances: List[List[float]]


class PredictionResponse(BaseModel):
    """Model predictions for each input batch."""

    predictions: List[float]


def _predict_one(features: List[float]) -> float:
    """Simple linear model with sigmoid activation."""

    coeffs = MODEL.get("entry_coefficients", [])
    intercept = MODEL.get("entry_intercept", 0.0)
    if len(coeffs) != len(FEATURE_NAMES):
        raise ValueError("model definition inconsistent")
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("feature length mismatch")
    score = sum(c * f for c, f in zip(coeffs, features)) + intercept
    return 1.0 / (1.0 + math.exp(-score))


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """Return predictions for a batch of feature vectors."""

    results: List[float] = []
    for features in req.instances:
        try:
            results.append(_predict_one(features))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(predictions=results)


def main() -> None:
    """Run a small FastAPI service that serves the trained model."""

    parser = argparse.ArgumentParser(description="Serve the distilled model")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    uvicorn.run("botcopier.scripts.serve_model:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
