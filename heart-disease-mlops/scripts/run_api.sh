#!/usr/bin/env bash
# Run FastAPI from repo root without remembering PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}" "$@"
