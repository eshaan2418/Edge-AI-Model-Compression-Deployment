from __future__ import annotations

from typing import Any

import torch.nn as nn

from edge_ai_compression.core.experiment import CompressionConfig
from edge_ai_compression.core.pipeline import CompressionPipeline


class SequentialCompression:
    """Explicit alias for the ordered compression pipeline (distill → prune → quant)."""

    def __init__(self, config: CompressionConfig) -> None:
        self._pipeline = CompressionPipeline(config)

    def run(
        self,
        model: nn.Module,
        data: Any,
        *,
        data_dir: str,
        batch_size: int,
        device: str,
    ) -> nn.Module:
        return self._pipeline.run(
            model, data, data_dir=data_dir, batch_size=batch_size, device=device
        )
