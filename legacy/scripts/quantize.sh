#!/usr/bin/env bash
set -euo pipefail

python -m legacy.model_compression.cli quantize --checkpoint models/baseline_resnet18.pt --output models/quantized_dynamic.pt
