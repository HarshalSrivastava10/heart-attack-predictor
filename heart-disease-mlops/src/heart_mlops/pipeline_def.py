from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops.config import FEATURE_COLUMNS

# Continuous measurements — scale after imputation
_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Discrete / categorical — one-hot after mode imputation
_CATEGORICAL = [c for c in FEATURE_COLUMNS if c not in _NUMERIC]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, _NUMERIC),
            ("cat", categorical_pipe, _CATEGORICAL),
        ]
    )


def build_full_pipeline(model_name: str = "logistic") -> Pipeline:
    """Classification pipeline: preprocess + estimator."""
    preprocessor = build_preprocessor()
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError("model_name must be 'logistic' or 'random_forest'")
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
