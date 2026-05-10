#!/usr/bin/env bash
# Regenerate reports/MLOps_Assignment_Report.docx from the Markdown source (Pandoc).
# Run from repo root: ./scripts/docx_report.sh
#
# Optional for embedded SVG diagrams in Word output: install librsvg so rsvg-convert exists:
#   brew install librsvg   # macOS
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REP="${ROOT}/reports"
MD="${REP}/MLOps_Assignment_Report.md"
DOCX="${REP}/MLOps_Assignment_Report.docx"

pandoc "${MD}" -o "${DOCX}" --standalone --toc --resource-path="${REP}:${REP}/diagrams"
echo "Wrote ${DOCX}"
