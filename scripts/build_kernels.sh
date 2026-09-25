#!/usr/bin/env bash
# Build the C++ kernel extension into edge_ai_compression/inference/.
# Needs: pip install -e ".[kernels]"  (cmake, ninja, nanobind)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
cmake -S csrc -B build/csrc -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build build/csrc
cmake --install build/csrc
