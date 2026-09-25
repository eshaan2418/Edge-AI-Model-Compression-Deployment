#!/usr/bin/env bash
# Regenerate every figure and table from the experiment DB into results/figures/.
# Analyses whose inputs do not exist yet are listed as PENDING in MANIFEST.md.
set -euo pipefail
cd "$(dirname "$0")"
python -m edge_ai_compression.analysis.reproduce --results "${1:-results}" --out "${2:-results/figures}"
