#!/usr/bin/env bash
# Build paper/main.pdf (needs latexmk + a TeX distribution).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
