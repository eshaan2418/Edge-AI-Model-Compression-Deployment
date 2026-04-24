from __future__ import annotations

from typing import Any

import torch.nn as nn

from edge_ai_compression.benchmarking.latency_profiler import profile_model_on_tensor
from edge_ai_compression.hardware.device_runner import DeviceRunner


class CpuRunner(DeviceRunner):
    def __init__(self) -> None:
        super().__init__("cpu")

    def _init_device(self, device_type: str) -> str:
        return "cpu"

    def run_inference(self, model: nn.Module, input_data: Any) -> dict[str, float]:
        model = model.to(self.device)
        sample = input_data.to(self.device)
        return profile_model_on_tensor(model, sample, warmup=3, repeats=100)
