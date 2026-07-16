#!/usr/bin/env python3
"""Root entry point for the offline compression demo.

Thin wrapper so the demo can be run as either:

    python demo.py --quick
    python -m edge_ai_compression.demo --quick
"""

from edge_ai_compression.demo import main

if __name__ == "__main__":
    main()
