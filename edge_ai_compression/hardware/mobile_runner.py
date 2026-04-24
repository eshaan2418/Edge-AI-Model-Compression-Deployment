from __future__ import annotations

"""Mobile / NPU execution is device-specific; this module centralizes placeholders for CoreML/TFLite paths."""


class MobileRunner:
    def __init__(self, backend: str = "placeholder") -> None:
        self.backend = backend

    def notes(self) -> str:
        return (
            f"Mobile backend '{self.backend}': integrate CoreML Tools or TFLite converter "
            "for production mobile benchmarking."
        )
