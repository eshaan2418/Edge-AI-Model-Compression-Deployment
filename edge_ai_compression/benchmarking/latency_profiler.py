from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def profile_latency_ms(
    forward_fn: Callable[[], None],
    *,
    warmup: int = 5,
    repeats: int = 200,
) -> dict[str, Any]:
    trace = profile_latency_trace(forward_fn, warmup=warmup, repeats=repeats)
    arr = np.asarray(trace, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "trace_ms": trace,
    }


def profile_latency_trace(
    forward_fn: Callable[[], None],
    *,
    warmup: int = 5,
    repeats: int = 200,
) -> list[float]:
    for _ in range(warmup):
        forward_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def profile_model_detailed(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 200,
) -> dict[str, Any]:
    model.eval()

    def one() -> None:
        with torch.no_grad():
            _ = model(sample)

    t_cold0 = time.perf_counter()
    with torch.no_grad():
        _ = model(sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t_cold0) * 1000.0

    trace = profile_latency_trace(one, warmup=warmup, repeats=repeats)
    arr = np.asarray(trace, dtype=np.float64)
    return {
        "cold_start_ms": float(cold_ms),
        "warm_mean_ms": float(np.mean(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "trace_ms": trace,
        "throughput_ips": float(1000.0 / max(float(np.mean(arr)), 1e-6)),
    }


def profile_model_on_tensor(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 200,
) -> dict[str, float]:
    out = profile_model_detailed(model, sample, warmup=warmup, repeats=repeats)
    return {k: float(v) for k, v in out.items() if k != "trace_ms" and isinstance(v, (int, float))}
