"""Process memory high-water mark (see docs/DECISIONS.md D1.8)."""

from __future__ import annotations

import resource
import sys

MIB = 1024 * 1024


def peak_rss_bytes() -> int:
    """Resident-set high-water mark of this process.

    ``ru_maxrss`` is bytes on macOS and KiB on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024
