#!/usr/bin/env python3
"""Train PyTorch baseline (CIFAR); wraps ``src.model_compression.train``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_compression.train import main  # noqa: E402


if __name__ == "__main__":
    main()
