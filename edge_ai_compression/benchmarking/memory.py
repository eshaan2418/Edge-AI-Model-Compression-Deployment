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

    Linux: /proc VmHWM. ``ru_maxrss`` is unusable there for spawned workers: at
    execve the kernel folds the pre-exec address space's peak (a fork of the
    parent) into the process's maxrss, so a worker spawned from a 1 GB pytest
    process reports ~1 GB from its first instruction (observed in CI; DECISIONS
    D1.15). VmHWM belongs to the new address space and starts fresh at exec.
    macOS: ``ru_maxrss`` (bytes), which does not inherit the parent's peak.
    """
    src = memory_sources()
    hwm = src["vm_hwm"]
    return hwm if hwm is not None else int(src["ru_maxrss"] or 0)
