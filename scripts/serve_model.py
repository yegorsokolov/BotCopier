import json
import math
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model.json at startup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.json"
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    MODEL = json.load(f)

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
    if len(features) != len(coeffs):
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
