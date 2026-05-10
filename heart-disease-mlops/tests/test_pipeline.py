from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_score

from heart_mlops.data import load_cleveland_df
from heart_mlops.pipeline_def import build_full_pipeline


def test_cv_better_than_chance():
    df = load_cleveland_df()
    X = df.drop(columns=["target"])
    y = (df["target"] > 0).astype(int)
    pipe = build_full_pipeline("logistic")
    scores = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")
    assert np.mean(scores) > 0.55
