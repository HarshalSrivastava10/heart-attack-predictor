from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heart_mlops.config import FEATURE_COLUMNS, TARGET_COLUMN


def load_cleveland_df(path: Path | None = None) -> pd.DataFrame:
    """Load processed Cleveland data; `?` marks missing values."""
    from heart_mlops.config import DATA_RAW

    csv_path = path or DATA_RAW
    df = pd.read_csv(
        csv_path,
        header=None,
        names=FEATURE_COLUMNS + [TARGET_COLUMN],
        na_values=["?"],
    )
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Presence of disease: original target > 0 -> 1."""
    out = df.copy()
    out["target_binary"] = (out[TARGET_COLUMN] > 0).astype(np.int64)
    return out


def train_val_split_xy(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return X, y (binary) and stratified train/test indices via sklearn."""
    from sklearn.model_selection import train_test_split

    df_b = add_binary_target(df)
    X = df_b[FEATURE_COLUMNS]
    y = df_b["target_binary"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
