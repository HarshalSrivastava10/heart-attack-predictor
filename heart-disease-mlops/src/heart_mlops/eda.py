"""EDA plots: histograms, correlation heatmap, class balance (assignment Task 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_mlops.config import EDA_DIR
from heart_mlops.data import add_binary_target, load_cleveland_df


def run_eda(out_dir: Path | None = None) -> None:
    out = out_dir or EDA_DIR
    out.mkdir(parents=True, exist_ok=True)

    df = load_cleveland_df()
    df = add_binary_target(df)

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    df["target_binary"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4c72b0", "#dd8452"]
    )
    ax.set_title("Class balance (binary: disease vs no disease)")
    ax.set_xlabel("target_binary (0=no, 1=yes)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=120)
    plt.close(fig)

    # Histograms for numeric columns
    numeric_cols = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()
    for i, col in enumerate(numeric_cols):
        df[col].dropna().hist(ax=axes[i], bins=20, color="#4c72b0", alpha=0.85)
        axes[i].set_title(col)
    axes[-1].axis("off")
    fig.suptitle("Feature histograms (numeric)")
    fig.tight_layout()
    fig.savefig(out / "histograms_numeric.png", dpi=120)
    plt.close(fig)

    # Correlation heatmap (numeric subset + binary target)
    corr_cols = numeric_cols + ["target_binary"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"EDA figures written to {out.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_eda(args.out)


if __name__ == "__main__":
    main()
