from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class DeviceRunner(ABC):
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.device = self._init_device(device_type)

    @abstractmethod
    def _init_device(self, device_type: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, model: nn.Module, input_data: Any) -> Any:
        raise NotImplementedError
