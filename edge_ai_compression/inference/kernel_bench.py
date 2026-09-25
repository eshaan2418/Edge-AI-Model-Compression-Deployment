"""Kernel-level benchmarks (same protocol as model benchmarks: fresh process per repeat).

A ``KernelSpec`` names one GEMM (or machine-peak microbenchmark) and a shape. The
worker (``inference.kernel_worker``) builds seeded operands, packs the weights
outside the timed region, and times only the kernel call. Results feed the
roofline and sparsity analyses and are logged to ``kernel_benchmarks.csv``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.isolation import spawn_worker
from edge_ai_compression.benchmarking.stats import Summary, bootstrap_ci, summarize

GEMM_OPS = ("gemm_f32", "gemm_s8", "gemm_w4", "gemm_sparse24", "gemm_csr")
PEAK_OPS = ("peak_fma_f32", "peak_dot_s8", "triad")
PEAK_UNITS = {"peak_fma_f32": "GFLOP/s", "peak_dot_s8": "GOP/s", "triad": "GB/s"}


@dataclass(frozen=True)
class KernelSpec:
    op: str
    m: int = 0
    n: int = 0
    k: int = 0
    sparsity: float = 0.0  # gemm_csr only (2:4 is fixed at 0.5)
    isa: str = "auto"
    group: int = 32  # gemm_w4 only
    seed: int = 0

    def __post_init__(self) -> None:
        if self.op not in GEMM_OPS + PEAK_OPS:
            raise ValueError(f"unknown kernel op '{self.op}'")
        if self.op in GEMM_OPS and min(self.m, self.n, self.k) < 1:
            raise ValueError(f"{self.op} needs positive m, n, k")
        if self.op == "gemm_sparse24" and self.k % 4:
            raise ValueError("gemm_sparse24 needs k divisible by 4")
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("sparsity must be in [0, 1]")

    @property
    def is_peak(self) -> bool:
        return self.op in PEAK_OPS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KernelReport:
    """Aggregated over processes. GEMM ops: latency (microseconds) and achieved
    GFLOP/s (useful FLOPs, i.e. nonzeros only for sparse ops). Peak ops:
    ``achieved`` in ``achieved_unit`` (median over processes), no latency."""

    spec: KernelSpec
    config: BenchmarkConfig
    isa: str
    flops: int
    bytes: int
    weight_bytes: int
    latency_us: Summary | None
    process_medians_us: tuple[float, ...]
    latency_median_us: float | None
    latency_median_ci_us: tuple[float, float] | None
    achieved: float
    achieved_unit: str

    @property
    def arithmetic_intensity(self) -> float | None:
        return self.flops / self.bytes if self.bytes else None


def benchmark_kernel(spec: KernelSpec, cfg: BenchmarkConfig) -> KernelReport:
    outs = [
        spawn_worker(
            "edge_ai_compression.inference.kernel_worker",
            {"spec": spec.to_dict(), "config": cfg.to_dict()},
        )
        for _ in range(cfg.process_repeats)
    ]
    first = outs[0]
    if spec.is_peak:
        vals = [float(o["achieved"]) for o in outs]
        return KernelReport(
            spec=spec,
            config=cfg,
            isa=first["isa"],
            flops=0,
            bytes=0,
            weight_bytes=0,
            latency_us=None,
            process_medians_us=(),
            latency_median_us=None,
            latency_median_ci_us=None,
            achieved=float(np.median(vals)),
            achieved_unit=PEAK_UNITS[spec.op],
        )
    traces_us = [np.asarray(o["trace_ns"], dtype=np.float64) / 1e3 for o in outs]
    medians = tuple(float(np.median(t)) for t in traces_us)
    median = float(np.median(medians))
    return KernelReport(
        spec=spec,
        config=cfg,
        isa=first["isa"],
        flops=int(first["flops"]),
        bytes=int(first["bytes"]),
        weight_bytes=int(first["weight_bytes"]),
        latency_us=summarize(np.concatenate(traces_us)),
        process_medians_us=medians,
        latency_median_us=median,
        latency_median_ci_us=bootstrap_ci(medians) if len(medians) > 1 else None,
        achieved=int(first["flops"]) / (median * 1e-6) / 1e9,
        achieved_unit="GFLOP/s",
    )
