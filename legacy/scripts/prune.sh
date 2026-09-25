#!/usr/bin/env bash
set -euo pipefail

python -m legacy.model_compression.cli prune --checkpoint models/baseline_resnet18.pt --amount 0.5 --output models/pruned.pt
