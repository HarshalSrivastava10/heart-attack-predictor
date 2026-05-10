#!/usr/bin/env bash
# Regenerate reports/MLOps_Assignment_Report.html and .pdf (Pandoc + Chrome).
# Run from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REP="${ROOT}/reports"
MD="${REP}/MLOps_Assignment_Report.md"
HTML="${REP}/MLOps_Assignment_Report.html"
PDF="${REP}/MLOps_Assignment_Report.pdf"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

cd "${REP}"
pandoc "${MD}" -o "${HTML}" --standalone --toc --embed-resources --resource-path=".:diagrams"

if [[ -x "${CHROME}" ]]; then
  "${CHROME}" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="${PDF}" "file://${HTML}"
  echo "Wrote ${PDF}"
else
  echo "Chrome not found at ${CHROME}; open ${HTML} and Print to PDF manually."
fi
