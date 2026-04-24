from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.benchmarking.evaluator import Evaluator
from edge_ai_compression.core.registry import ModelRegistry


def test_evaluator_runs():
    m = ModelRegistry.create("resnet18_cifar")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    ev = Evaluator(device="cpu", latency_repeats=5, latency_warmup=1)
    r = ev.evaluate(m, loader)
    assert 0.0 <= r.accuracy <= 1.0
    assert r.latency_ms_mean > 0
