from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from heart_mlops.data import load_cleveland_df, train_val_split_xy
from heart_mlops.pipeline_def import build_full_pipeline


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_features():
    """First row of Cleveland processed data (known valid)."""
    return {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0,
    }


@pytest.fixture
def tmp_model_bundle(tmp_path, monkeypatch):
    """Train a tiny pipeline and point MODEL_PATH at it for API tests."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "bundle.joblib"))
    df = load_cleveland_df()
    X_train, _, y_train, _ = train_val_split_xy(df, test_size=0.3)
    pipe = build_full_pipeline("logistic")
    pipe.fit(X_train, y_train)
    bundle = {
        "pipeline": pipe,
        "model_name": "logistic",
        "feature_columns": list(X_train.columns),
        "roc_auc_holdout": 0.9,
        "cv_splits": 3,
    }
    joblib.dump(bundle, tmp_path / "bundle.joblib")

    import api.main as api_main

    api_main.reset_model_cache()

    yield tmp_path / "bundle.joblib"

    api_main.reset_model_cache()
    monkeypatch.delenv("MODEL_PATH", raising=False)
