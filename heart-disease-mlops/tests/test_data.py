from __future__ import annotations

from heart_mlops.config import DATA_RAW, FEATURE_COLUMNS
from heart_mlops.data import add_binary_target, load_cleveland_df


def test_load_cleveland_shape():
    df = load_cleveland_df(DATA_RAW)
    assert len(df) == 303
    assert list(df.columns) == FEATURE_COLUMNS + ["target"]


def test_binary_target_range():
    df = add_binary_target(load_cleveland_df())
    assert set(df["target_binary"].unique()).issubset({0, 1})


def test_no_missing_after_preprocess_pipeline(sample_features):
    """Pipeline imputes — single-row frame should not error."""
    from heart_mlops.pipeline_def import build_full_pipeline

    df = load_cleveland_df()
    X = df[FEATURE_COLUMNS].head(20)
    y = (df["target"] > 0).astype(int).head(20)
    pipe = build_full_pipeline("random_forest")
    pipe.fit(X, y)
    row = {k: sample_features[k] for k in FEATURE_COLUMNS}
    import pandas as pd

    X_one = pd.DataFrame([row])
    pred = pipe.predict(X_one)
    assert pred.shape == (1,)
