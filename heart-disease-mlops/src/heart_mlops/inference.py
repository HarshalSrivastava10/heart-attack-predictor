from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from heart_mlops.config import FEATURE_COLUMNS, MODELS_DIR


def resolve_model_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    return MODELS_DIR / "heart_classifier.joblib"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    p = resolve_model_path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {p}. Run: python -m heart_mlops.train"
        )
    return joblib.load(p)


def risk_level_from_probability(p_positive: float) -> str:
    """
    Map P(disease) to a plain-language band for API responses.

    Bands (on probability of the positive class):
    - no risk:    [0.00, 0.25)
    - low risk:   [0.25, 0.50)
    - medium risk:[0.50, 0.75)
    - high risk:  [0.75, 1.00]
    """
    if p_positive < 0.25:
        return "no risk"
    if p_positive < 0.50:
        return "low risk"
    if p_positive < 0.75:
        return "medium risk"
    return "high risk"


def predict_proba_dict(
    bundle: dict[str, Any],
    features: dict[str, float],
) -> tuple[int, float, float]:
    """
    Returns (predicted_class 0/1, probability_positive, confidence).
    confidence = max(p, 1-p).
    """
    pipe = bundle["pipeline"]
    row = pd.DataFrame([{k: features[k] for k in FEATURE_COLUMNS}])
    proba = pipe.predict_proba(row)[0]
    p_pos = float(proba[1])
    pred = int(p_pos >= 0.5)
    confidence = float(max(p_pos, 1.0 - p_pos))
    return pred, p_pos, confidence
