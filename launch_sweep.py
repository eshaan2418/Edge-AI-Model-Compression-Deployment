#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_ai_compression.experiments.launch_sweep import main  # noqa: E402


if __name__ == "__main__":
    main()
