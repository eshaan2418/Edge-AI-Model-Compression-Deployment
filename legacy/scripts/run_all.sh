#!/usr/bin/env bash
set -euo pipefail

legacy/scripts/train_baseline.sh
legacy/scripts/prune.sh
legacy/scripts/quantize.sh
legacy/scripts/distill.sh
