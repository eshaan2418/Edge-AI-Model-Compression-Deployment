#!/usr/bin/env bash
set -euo pipefail

python -m legacy.model_compression.cli distill --teacher models/baseline_resnet18.pt
