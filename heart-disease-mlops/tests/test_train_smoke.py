"""Smoke test training entrypoint (fast mode) — optional heavy test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_train_fast_smoke(project_root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(project_root)
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = str(tmp_path / "mlruns")
    cp = project_root / "data" / "raw" / "processed.cleveland.data"
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "heart_mlops.train",
            "--fast",
            "--data",
            str(cp),
        ],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr + rc.stdout
    assert (project_root / "models" / "heart_classifier.joblib").is_file()
