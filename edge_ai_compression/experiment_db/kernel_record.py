"""Rows of ``kernel_benchmarks.csv`` (kernel-level and machine-peak measurements)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any

from edge_ai_compression.benchmarking.fingerprint import Fingerprint
from edge_ai_compression.inference.kernel_bench import KernelReport

KERNEL_SCHEMA_VERSION = 1


@dataclass
class KernelRecord:
    """Latency columns are microseconds; empty for machine-peak rows."""

    run_id: str
    timestamp: str
    schema_version: int
    study: str
    git_commit: str | None
    git_dirty: bool | None
    fingerprint_hash: str
    op: str
    isa: str
    m: int
    n: int
    k: int
    sparsity: float
    group: int
    weight_bytes: int
    flops: int
    bytes: int
    arithmetic_intensity: float | None
    n_iters: int
    process_repeats: int
    latency_median_us: float | None
    latency_median_ci_lo_us: float | None
    latency_median_ci_hi_us: float | None
    latency_p50_us: float | None
    latency_p95_us: float | None
    latency_p99_us: float | None
    latency_max_us: float | None
    achieved: float
    achieved_unit: str

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


KERNEL_CSV_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(KernelRecord))


def kernel_record(report: KernelReport, fingerprint: Fingerprint, study: str) -> KernelRecord:
    lat, ci, spec = report.latency_us, report.latency_median_ci_us, report.spec
    return KernelRecord(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        schema_version=KERNEL_SCHEMA_VERSION,
        study=study,
        git_commit=fingerprint.git_commit,
        git_dirty=fingerprint.git_dirty,
        fingerprint_hash=fingerprint.hash,
        op=spec.op,
        isa=report.isa,
        m=spec.m,
        n=spec.n,
        k=spec.k,
        sparsity=spec.sparsity
        if spec.op == "gemm_csr"
        else (0.5 if spec.op == "gemm_sparse24" else 0.0),
        group=spec.group if spec.op == "gemm_w4" else 0,
        weight_bytes=report.weight_bytes,
        flops=report.flops,
        bytes=report.bytes,
        arithmetic_intensity=report.arithmetic_intensity,
        n_iters=lat.n if lat else 0,
        process_repeats=report.config.process_repeats,
        latency_median_us=report.latency_median_us,
        latency_median_ci_lo_us=ci[0] if ci else None,
        latency_median_ci_hi_us=ci[1] if ci else None,
        latency_p50_us=lat.p50 if lat else None,
        latency_p95_us=lat.p95 if lat else None,
        latency_p99_us=lat.p99 if lat else None,
        latency_max_us=lat.max if lat else None,
        achieved=report.achieved,
        achieved_unit=report.achieved_unit,
    )
