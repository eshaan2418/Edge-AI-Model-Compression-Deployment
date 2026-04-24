from __future__ import annotations

from typing import Callable

import psutil


def estimate_peak_rss_mib(run: Callable[[], None]) -> float:
    process = psutil.Process()
    before = process.memory_info().rss
    run()
    after = process.memory_info().rss
    peak = max(before, after)
    return peak / (1024 * 1024)
