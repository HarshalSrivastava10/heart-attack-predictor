"""
Train two classifiers with CV, MLflow tracking, save best sklearn pipeline as joblib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from heart_mlops.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJECT_ROOT,
)
from heart_mlops.data import load_cleveland_df, train_val_split_xy
from heart_mlops.pipeline_def import build_full_pipeline


def _evaluate_holdout(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, proba)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }
    return metrics, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer CV splits for CI/quick runs",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override path to processed.cleveland.data",
    )
    args = parser.parse_args()

    cv_splits = 3 if args.fast else 5
    random_state = 42

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("heart-disease-cleveland")

    df = load_cleveland_df(args.data)
    X_train, X_test, y_train, y_test = train_val_split_xy(df)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_name = None
    best_auc = -1.0
    best_pipe = None

    for name in ("logistic", "random_forest"):
        with mlflow.start_run(run_name=name):
            pipe = build_full_pipeline(name)
            mlflow.log_param("model", name)
            mlflow.log_param("cv_splits", cv_splits)
            mlflow.log_param("n_samples_train", len(X_train))

            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision_macro",
                    "recall": "recall_macro",
                    "roc_auc": "roc_auc",
                },
                n_jobs=1,
            )
            for k, arr in scores.items():
                if k.startswith("test_"):
                    mlflow.log_metric(f"cv_{k}", float(np.mean(arr)))

            pipe.fit(X_train, y_train)
            holdout, proba_holdout = _evaluate_holdout(pipe, X_test, y_test)
            for k, v in holdout.items():
                if k != "classification_report":
                    mlflow.log_metric(f"holdout_{k}", v)

            mlflow.log_text(holdout["classification_report"], "holdout_report.txt")

            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(
                y_test, proba_holdout, ax=ax, name=name
            )
            fig.tight_layout()
            roc_png = ARTIFACTS_DIR / f"roc_curve_{name}.png"
            fig.savefig(roc_png, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(roc_png))

            example_X = X_train.head(1)
            example_out = pipe.predict_proba(example_X)
            sig = infer_signature(example_X, example_out)
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=sig,
                input_example=example_X,
            )

            auc = holdout["roc_auc"]
            if auc > best_auc:
                best_auc = auc
                best_name = name
                best_pipe = pipe

    assert best_pipe is not None
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "roc_auc_holdout": best_auc,
        "cv_splits": cv_splits,
    }
    out_path = MODELS_DIR / "heart_classifier.joblib"
    joblib.dump(bundle, out_path)

    meta = {
        "chosen_model": best_name,
        "holdout_roc_auc": best_auc,
        "artifact": str(out_path.relative_to(PROJECT_ROOT)),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }
    with open(MODELS_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
