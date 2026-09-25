"""Timing loop for a single process. Cross-process repetition lives in ``isolation``."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable

import numpy as np

from edge_ai_compression.benchmarking.config import BenchmarkConfig

Clock = Callable[[], int]


def _run_until(
    fn: Callable[[], None], min_iters: int, min_seconds: float, clock: Clock
) -> list[int]:
    samples: list[int] = []
    min_ns = int(min_seconds * 1e9)
    start = last = clock()
    while len(samples) < min_iters or last - start < min_ns:
        t0 = clock()
        fn()
        last = clock()
        samples.append(last - t0)
    return samples


def measure_latency(
    fn: Callable[[], None],
    cfg: BenchmarkConfig,
    *,
    clock: Clock = time.perf_counter_ns,
) -> np.ndarray:
    """Warm up, then time ``fn`` per call. Returns int64 nanoseconds, one per timed call.

    The garbage collector is disabled for the duration (as ``timeit`` does) so
    collection pauses are not attributed to the model.
    """
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        _run_until(fn, cfg.warmup_iters, cfg.min_warmup_s, clock)
        return np.asarray(_run_until(fn, cfg.iters, cfg.min_time_s, clock), dtype=np.int64)
    finally:
        if gc_was_enabled:
            gc.enable()


def apply_cpu_affinity(cores: tuple[int, ...] | None) -> None:
    """Pin the current process to ``cores`` (Linux only). macOS exposes no affinity API."""
    if cores is None:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise ValueError("benchmark.cpu_affinity is not supported on this platform")
    os.sched_setaffinity(0, set(cores))
