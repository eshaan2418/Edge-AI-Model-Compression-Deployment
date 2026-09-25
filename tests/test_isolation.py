from __future__ import annotations

import subprocess
import sys

import pytest
import torch.nn as nn

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.isolation import _worker_env, benchmark_model
from edge_ai_compression.core.experiment import (
    CompressionConfig,
    PruningSection,
    QuantizationSection,
)
from edge_ai_compression.core.pipeline import CompressionPipeline
from edge_ai_compression.core.registry import ModelRegistry

TINY = BenchmarkConfig(warmup_iters=1, min_warmup_s=0.0, iters=5, process_repeats=1, num_threads=1)
STAGES = ("startup", "import", "load", "first_inference", "cold_start")


def test_peak_rss_tracks_a_known_allocation():
    code = (
        "import numpy as np\n"
        "from edge_ai_compression.benchmarking.memory import MIB, memory_sources, peak_rss_bytes\n"
        "before, src0 = peak_rss_bytes(), memory_sources()\n"
        "a = np.ones(256 * MIB, dtype=np.uint8)\n"
        "print((peak_rss_bytes() - before) / MIB, src0, memory_sources())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=_worker_env()
    )
    delta = float(out.stdout.split()[0])
    assert 240 <= delta <= 300, out.stdout


def test_isolated_run_reports_trace_stages_and_memory():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    runs = benchmark_model(model, (2, 3, 8, 8), TINY)
    assert len(runs) == 1
    r = runs[0]
    assert r.trace_ns.shape == (5,) and (r.trace_ns > 0).all()
    assert set(r.stages_ms) == set(STAGES)
    assert all(r.stages_ms[s] > 0 for s in STAGES)
    parts = sum(r.stages_ms[s] for s in STAGES if s != "cold_start")
    assert r.stages_ms["cold_start"] == pytest.approx(parts, rel=1e-6)
    assert r.num_threads == 1
    assert r.peak_rss_mib > r.model_peak_rss_mib >= 0
    assert r.energy_j_per_inf is None


def test_process_repeats_are_independent_processes():
    model = nn.Linear(4, 2)
    cfg = BenchmarkConfig(warmup_iters=0, min_warmup_s=0.0, iters=2, process_repeats=2)
    runs = benchmark_model(model, (1, 4), cfg)
    assert len(runs) == 2


def test_compressed_resnet_roundtrips_through_worker():
    # Pruned + quantized (simulated) models must survive torch.save -> fresh-process load.
    m = ModelRegistry.create("resnet18_cifar")
    cfg = CompressionConfig(
        pruning=PruningSection(enabled=True, amount=0.3),
        quantization=QuantizationSection(enabled=True, act_bits=None),
    )
    m = CompressionPipeline(cfg, order=["prune", "quantize"]).run(
        m, None, data_dir="data", batch_size=1, device="cpu"
    )
    r = benchmark_model(m, (1, 3, 32, 32), TINY)[0]
    assert r.trace_ns.shape == (5,)


def test_worker_failure_surfaces_stderr(tmp_path):
    from edge_ai_compression.benchmarking.isolation import run_isolated

    bad = tmp_path / "missing.pt"
    with pytest.raises(RuntimeError, match="benchmark worker failed"):
        run_isolated(bad, (1, 4), TINY)


def test_worker_does_not_import_torch_before_timing_starts():
    code = (
        "import sys\n"
        "import edge_ai_compression.benchmarking.worker\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=_worker_env()
    )
    assert out.stdout.strip() == "False"
