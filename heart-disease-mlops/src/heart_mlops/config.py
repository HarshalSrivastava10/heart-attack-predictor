import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDA_DIR = ARTIFACTS_DIR / "eda"
_default_mlruns = str(PROJECT_ROOT / "mlruns")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", _default_mlruns)

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"
