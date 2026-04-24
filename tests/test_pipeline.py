from __future__ import annotations

import torch

from edge_ai_compression.core.experiment import CompressionConfig, PruningSection
from edge_ai_compression.core.pipeline import CompressionPipeline
from edge_ai_compression.core.registry import ModelRegistry


def test_compression_pipeline_prune_only():
    m = ModelRegistry.create("resnet18_cifar")
    cfg = CompressionConfig(pruning=PruningSection(enabled=True, amount=0.2))
    pipe = CompressionPipeline(cfg)
    out = pipe.run(m, None, data_dir="data", batch_size=4, device="cpu")
    assert isinstance(out, torch.nn.Module)
