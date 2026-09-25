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

import numpy as np
import torch
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


def run_isolated(
    model_path: Path,
    input_shape: tuple[int, ...],
    cfg: BenchmarkConfig,
    *,
    timeout_s: float = 3600.0,
) -> ProcessResult:
    """Benchmark a ``torch.save``-d model object in a new Python process."""
    with tempfile.TemporaryDirectory() as tmp:
        request, result = Path(tmp) / "request.json", Path(tmp) / "result.json"
        t_spawn = time.monotonic_ns()
        request.write_text(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "input_shape": list(input_shape),
                    "config": cfg.to_dict(),
                    "t_spawn_ns": t_spawn,
                }
            )
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_ai_compression.benchmarking.worker",
                str(request),
                str(result),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_worker_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"benchmark worker failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        out = json.loads(result.read_text())
    return ProcessResult(
        trace_ns=np.asarray(out["trace_ns"], dtype=np.int64),
        stages_ms={k: float(v) for k, v in out["stages_ms"].items()},
        peak_rss_mib=float(out["peak_rss_mib"]),
        model_peak_rss_mib=float(out["model_peak_rss_mib"]),
        num_threads=int(out["num_threads"]),
    )


def benchmark_model(
    model: nn.Module, input_shape: tuple[int, ...], cfg: BenchmarkConfig
) -> list[ProcessResult]:
    """Save ``model`` once, then run ``cfg.process_repeats`` independent processes."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.cpu(), path)
        return [run_isolated(path, input_shape, cfg) for _ in range(cfg.process_repeats)]
