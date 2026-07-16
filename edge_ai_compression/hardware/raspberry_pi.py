from __future__ import annotations

"""Raspberry Pi deployment helpers (CPU inference, optional ONNX export hooks)."""


class RaspberryPiProfile:
    def __init__(self, threads: int = 4) -> None:
        self.threads = threads

    def describe(self) -> str:
        return (
            f"ARM CPU inference profile (threads={self.threads}); "
            "export to TorchScript/ONNX for edge serving."
        )
