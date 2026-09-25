"""Run benchmarks in fresh processes (see docs/DECISIONS.md D1.1, D1.10)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

import edge_ai_compression
from edge_ai_compression.benchmarking.config import BenchmarkConfig

PACKAGE_ROOT = Path(edge_ai_compression.__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProcessResult:
    """One fresh-process benchmark: warm latency trace, cold-start stages, memory."""

    trace_ns: np.ndarray
    stages_ms: dict[str, float]
    peak_rss_mib: float
    model_peak_rss_mib: float
    energy_j_per_inf: float | None
    num_threads: int

    @property
    def trace_ms(self) -> np.ndarray:
        return self.trace_ns / 1e6


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    return env


def spawn_worker(
    module: str, request: dict[str, Any], *, timeout_s: float = 3600.0
) -> dict[str, Any]:
    """Run ``python -m module REQUEST.json RESULT.json`` in a fresh process.

    ``request`` gets ``t_spawn_ns`` (monotonic, system-wide) added just before
    the spawn so the worker can report its startup time.
    """
    with tempfile.TemporaryDirectory() as tmp:
        req_path, res_path = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req_path.write_text(json.dumps({**request, "t_spawn_ns": time.monotonic_ns()}))
        proc = subprocess.run(
            [sys.executable, "-m", module, str(req_path), str(res_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_worker_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"benchmark worker failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        return json.loads(res_path.read_text())


def run_isolated(
    model_path: Path,
    input_shape: tuple[int, ...],
    cfg: BenchmarkConfig,
    *,
    timeout_s: float = 3600.0,
) -> ProcessResult:
    """Benchmark an exported artifact (see ``inference.backends``) in a new Python process."""
    out = spawn_worker(
        "edge_ai_compression.benchmarking.worker",
        {"model_path": str(model_path), "input_shape": list(input_shape), "config": cfg.to_dict()},
        timeout_s=timeout_s,
    )
    return ProcessResult(
        trace_ns=np.asarray(out["trace_ns"], dtype=np.int64),
        stages_ms={k: float(v) for k, v in out["stages_ms"].items()},
        peak_rss_mib=float(out["peak_rss_mib"]),
        model_peak_rss_mib=float(out["model_peak_rss_mib"]),
        energy_j_per_inf=out["energy_j_per_inf"],
        num_threads=int(out["num_threads"]),
    )


def benchmark_model(
    model: nn.Module, input_shape: tuple[int, ...], cfg: BenchmarkConfig
) -> list[ProcessResult]:
    """Export ``model`` for ``cfg.backend`` once, then run ``cfg.process_repeats`` processes."""
    from edge_ai_compression.inference.backends import get_backend

    with tempfile.TemporaryDirectory() as tmp:
        path = get_backend(cfg.backend).export(model, input_shape, Path(tmp))
        return [run_isolated(path, input_shape, cfg) for _ in range(cfg.process_repeats)]
