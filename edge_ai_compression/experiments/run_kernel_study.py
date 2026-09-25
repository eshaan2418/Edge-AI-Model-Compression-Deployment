"""Run a kernel-level study from YAML and log every measurement to the experiment DB.

Config keys:
  study: name recorded on every row
  results_dir: experiment DB directory (default results)
  benchmark: BenchmarkConfig section (backend is ignored; kernels are single-threaded)
  peaks: log machine peaks (peak_fma_f32, peak_dot_s8, triad) for roofline analysis
  isas: list of ISA names or [auto]
  shapes: explicit [[m, n, k], ...] and/or
  shapes_from_model: {model: <registry name>, input_shape: [C, H, W]}  (conv/linear GEMM shapes)
  ops: list of op names, or {op: gemm_csr, sparsity: [..]} entries
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.fingerprint import check_environment, collect
from edge_ai_compression.experiment_db.kernel_record import kernel_record
from edge_ai_compression.experiment_db.writer import append_kernel_row
from edge_ai_compression.inference.kernel_bench import PEAK_OPS, KernelSpec, benchmark_kernel
from edge_ai_compression.utils.config_loader import load_yaml

KEYS = {"study", "results_dir", "benchmark", "peaks", "isas", "shapes", "shapes_from_model", "ops"}


def model_gemm_shapes(name: str, input_shape: list[int]) -> list[tuple[int, int, int]]:
    """Distinct (M, N, K) of every conv/linear layer, via the engine's profiler."""
    from edge_ai_compression.core.registry import ModelRegistry
    from edge_ai_compression.inference.engine import compile_model

    model = ModelRegistry.create(name)
    _, records = compile_model(model, "f32").profile(np.zeros(input_shape, np.float32))
    seen: list[tuple[int, int, int]] = []
    for r in records:
        if r.kind == "gemm" and (r.m, r.n, r.k) not in seen:
            seen.append((r.m, r.n, r.k))
    return seen


def build_specs(cfg: dict[str, Any]) -> list[KernelSpec]:
    shapes = [tuple(int(v) for v in s) for s in cfg.get("shapes", [])]
    if "shapes_from_model" in cfg:
        src = cfg["shapes_from_model"]
        shapes += [
            s for s in model_gemm_shapes(src["model"], src["input_shape"]) if s not in shapes
        ]
    specs: list[KernelSpec] = []
    for isa in cfg.get("isas", ["auto"]):
        if cfg.get("peaks", False):
            specs += [KernelSpec(op, isa=isa) for op in PEAK_OPS]
        for entry in cfg.get("ops", []):
            op = entry if isinstance(entry, str) else entry["op"]
            sparsities = [0.0] if isinstance(entry, str) else entry.get("sparsity", [0.0])
            for m, n, k in shapes:
                if op == "gemm_sparse24" and k % 4:
                    continue  # 2:4 needs K divisible by 4 (e.g. a 3-channel 3x3 stem has K=27)
                for sp in sparsities:
                    specs.append(KernelSpec(op, m, n, k, sparsity=float(sp), isa=isa))
    return specs


def run_study(cfg: dict[str, Any]) -> Path:
    unknown = set(cfg) - KEYS
    if unknown:
        raise ValueError(f"unknown kernel-study keys: {sorted(unknown)}")
    bench = BenchmarkConfig.from_dict(dict(cfg.get("benchmark") or {}))
    results_dir = Path(cfg.get("results_dir", "results"))
    study = str(cfg["study"])
    fingerprint = collect()
    problems = check_environment(fingerprint, strict=bench.strict_environment)
    torch.set_num_threads(1)
    specs = build_specs(cfg)
    for i, spec in enumerate(specs, 1):
        report = benchmark_kernel(spec, bench)
        record = kernel_record(report, fingerprint, study)
        append_kernel_row(
            record,
            results_dir,
            {
                "process_medians_us": list(report.process_medians_us),
                "environment_problems": problems,
                "fingerprint_dynamic": fingerprint.dynamic,
            },
        )
        print(
            f"[{i}/{len(specs)}] {spec.op} {spec.m}x{spec.n}x{spec.k} sp={spec.sparsity} "
            f"-> {report.achieved:.2f} {report.achieved_unit}"
        )
    return results_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    run_study(load_yaml(args.config))


if __name__ == "__main__":
    main()
