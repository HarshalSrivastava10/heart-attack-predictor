"""
Heart disease prediction API (assignment Tasks 6 & 8).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from heart_mlops.config import FEATURE_COLUMNS
from heart_mlops.inference import (
    load_bundle,
    predict_proba_dict,
    risk_level_from_probability,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heart_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("API startup complete (Prometheus /metrics enabled).")
    yield


app = FastAPI(
    title="Heart Disease Classifier",
    description="Cleveland UCI — binary disease risk prediction",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

_bundle = None


def reset_model_cache() -> None:
    """Test hook: clear cached model so MODEL_PATH env is re-read."""
    global _bundle
    _bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        try:
            _bundle = load_bundle()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return _bundle


class PredictIn(BaseModel):
    """All 13 Cleveland features (see heart-disease.names)."""

    age: float = Field(..., ge=0, le=120)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=0, le=300)
    chol: float = Field(..., ge=0, le=600)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, le=250)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: float = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: float = Field(..., ge=3, le=7)


class PredictOut(BaseModel):
    prediction: int = Field(description="0 = no disease, 1 = disease")
    probability_positive: float
    confidence: float
    risk_level: str = Field(
        description=(
    "Plain-language band from probability "
    "(no / low / medium / high risk)"
)
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(
    body: PredictIn,
    x_request_id: Annotated[str | None, Header()] = None,
):
    start = time.perf_counter()
    feats = body.model_dump()
    missing = [k for k in FEATURE_COLUMNS if k not in feats]
    if missing:
        raise HTTPException(400, f"Missing keys: {missing}")

    bundle = get_bundle()
    pred, p_pos, conf = predict_proba_dict(bundle, feats)
    risk = risk_level_from_probability(p_pos)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict request_id=%s prediction=%s p_pos=%.4f risk=%s latency_ms=%.2f",
        x_request_id or "-",
        pred,
        p_pos,
        risk,
        elapsed_ms,
    )

    return PredictOut(
        prediction=pred,
        probability_positive=p_pos,
        confidence=conf,
        risk_level=risk,
    )
