"""Time one kernel in a fresh process (spawned by ``kernel_bench.benchmark_kernel``)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.timing import apply_cpu_affinity, measure_latency
from edge_ai_compression.inference import kernels, packing
from edge_ai_compression.inference.kernel_bench import KernelSpec


def _gemm(spec: KernelSpec, isa: str) -> tuple[Callable[[], None], dict[str, int]]:
    """Build seeded operands, pack outside the timed region, return (call, sizes)."""
    rng = np.random.default_rng(spec.seed)
    m, n, k = spec.m, spec.n, spec.k
    w = rng.standard_normal((m, k), dtype=np.float32)
    b32 = rng.standard_normal((k, n), dtype=np.float32)
    out32 = np.empty((m, n), np.float32)
    C = kernels.require()
    if spec.op == "gemm_s8":
        q, _ = packing.quantize_rows_s8(w)
        a = C.pack_s8(isa, q)
        b = rng.integers(-127, 128, size=(k, n), dtype=np.int8)
        out = np.empty((m, n), np.int32)
        call = lambda: C.gemm_s8(a, b, out)  # noqa: E731
        return call, {
            "flops": 2 * m * n * k,
            "weight_bytes": a.nbytes,
            "io_bytes": b.nbytes + out.nbytes,
        }
    if spec.op == "gemm_f32":
        a = C.pack_f32(w)
        flops = 2 * m * n * k
        call = lambda: C.gemm_f32(isa, a, b32, None, False, out32)  # noqa: E731
    elif spec.op == "gemm_w4":
        a = kernels.make_w4(*packing.quantize_w4(w, spec.group), k, spec.group)
        flops = 2 * m * n * k
        call = lambda: C.gemm_w4(isa, a, b32, None, False, out32)  # noqa: E731
    elif spec.op == "gemm_sparse24":
        a = kernels.make_sparse24(*packing.prune_24(w))
        flops = 2 * m * n * (k // 2)
        call = lambda: C.gemm_sparse24(isa, a, b32, None, False, out32)  # noqa: E731
    else:  # gemm_csr
        a = kernels.make_csr(*packing.to_csr(packing.magnitude_prune(w, spec.sparsity)), k)
        flops = 2 * int(a.nnz) * n
        call = lambda: C.gemm_csr(isa, a, b32, None, False, out32)  # noqa: E731
    return call, {"flops": flops, "weight_bytes": a.nbytes, "io_bytes": b32.nbytes + out32.nbytes}


def main(argv: list[str]) -> None:
    req = json.loads(Path(argv[0]).read_text())
    spec = KernelSpec(**req["spec"])
    cfg = BenchmarkConfig.from_dict(req["config"])
    apply_cpu_affinity(cfg.cpu_affinity)
    isa = kernels.best_isa() if spec.isa == "auto" else spec.isa
    C = kernels.require()
    result: dict[str, Any] = {"isa": isa}
    if spec.op == "peak_fma_f32":
        result["achieved"] = C.peak_fma_gflops(isa, cfg.min_time_s or 0.5)
    elif spec.op == "peak_dot_s8":
        result["achieved"] = C.peak_s8_gops(isa, cfg.min_time_s or 0.5)
    elif spec.op == "triad":
        # STREAM-style: best of a few passes over 3 x 256 MiB arrays.
        result["achieved"] = C.triad_gbps(256 * 1024 * 1024, min(max(cfg.iters, 3), 10))
    else:
        call, sizes = _gemm(spec, isa)
        trace = measure_latency(call, cfg)
        result.update(
            trace_ns=trace.tolist(),
            flops=sizes["flops"],
            weight_bytes=sizes["weight_bytes"],
            # Compulsory traffic: weights + activations in + outputs out, once each.
            bytes=sizes["weight_bytes"] + sizes["io_bytes"],
        )
    Path(argv[1]).write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
