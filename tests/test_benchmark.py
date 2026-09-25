from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.evaluator import Evaluator
from edge_ai_compression.core.registry import ModelRegistry

TINY = BenchmarkConfig(
    warmup_iters=1, min_warmup_s=0.0, iters=5, process_repeats=2, strict_environment=False
)


def test_evaluator_report():
    m = ModelRegistry.create("resnet18_cifar")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    r = Evaluator(device="cpu", config=TINY).evaluate(m, loader)
    assert 0.0 <= r.accuracy <= 1.0
    assert r.input_shape == (1, 3, 32, 32)
    assert r.latency.n == 10
    assert len(r.process_medians_ms) == 2
    lo, hi = r.latency_median_ci_ms
    assert lo <= r.latency_median_ms <= hi
    assert r.cold_start_ms > 0 and r.peak_rss_mib > 0 and r.size_mb > 0
    assert r.energy_j_per_inf is None
    assert len(r.fingerprint.hash) == 16
    assert set(r.to_dict()) >= {"latency_ms", "stages_ms", "fingerprint_hash"}


def test_single_process_has_no_ci():
    m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 2, 2), torch.zeros(4, dtype=torch.long)))
    cfg = BenchmarkConfig(
        warmup_iters=0, min_warmup_s=0.0, iters=3, process_repeats=1, strict_environment=False
    )
    r = Evaluator(device="cpu", config=cfg).evaluate(m, loader)
    assert r.latency_median_ci_ms is None
