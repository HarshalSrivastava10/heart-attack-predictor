#!/usr/bin/env python3
"""
Download processed Cleveland data from UCI (assignment: data acquisition).
Falls back to instructions if the URL changes.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

UCI_CLEVELAND = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {UCI_CLEVELAND} -> {dest}")
    urllib.request.urlretrieve(UCI_CLEVELAND, dest)
    print("Done.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "data" / "raw" / "processed.cleveland.data",
    )
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
