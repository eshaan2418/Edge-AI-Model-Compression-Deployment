from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    RASPBERRY_PI = "raspberry_pi"
    MOBILE = "mobile"


@dataclass
class DeviceSpec:
    device_type: DeviceType
    label: str = ""


class DeviceManager:
    def __init__(self, spec: DeviceSpec) -> None:
        self.spec = spec

    def torch_device(self) -> str:
        if self.spec.device_type == DeviceType.CUDA:
            return "cuda"
        return "cpu"
