"""Process memory high-water mark (see docs/DECISIONS.md D1.8)."""

from __future__ import annotations

import re
import resource
import sys
from pathlib import Path

MIB = 1024 * 1024
_STATUS = Path("/proc/self/status")


def _proc_status_kib(field: str) -> int | None:
    if not _STATUS.is_file():
        return None
    m = re.search(rf"^{field}:\s+(\d+)\s+kB", _STATUS.read_text(), re.MULTILINE)
    return int(m.group(1)) if m else None


def memory_sources() -> dict[str, int | None]:
    """Every available peak/current RSS reading, in bytes (for diagnostics)."""
    ru = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    hwm, rss = _proc_status_kib("VmHWM"), _proc_status_kib("VmRSS")
    return {
        # ru_maxrss is bytes on macOS and KiB on Linux.
        "ru_maxrss": ru if sys.platform == "darwin" else ru * 1024,
        "vm_hwm": hwm * 1024 if hwm is not None else None,
        "vm_rss": rss * 1024 if rss is not None else None,
    }


def peak_rss_bytes() -> int:
    """Resident-set high-water mark of this process.

    On Linux this is the max of ru_maxrss, /proc VmHWM, and current VmRSS: on
    some CI kernels ru_maxrss did not reflect a fresh 256 MiB allocation, so no
    single source is trusted.
    """
    return max(v for v in memory_sources().values() if v is not None)
