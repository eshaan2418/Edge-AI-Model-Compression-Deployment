from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.analysis.failure_analysis import FailureAnalyzer
from edge_ai_compression.core.registry import ModelRegistry


def test_failure_analyzer():
    m1 = ModelRegistry.create("resnet18_cifar")
    m2 = ModelRegistry.create("resnet18_cifar")
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    fa = FailureAnalyzer()
    stats = fa.compare(m1, m2, loader, "cpu")
    assert "compressed_error_rate" in stats
