"""Typed interfaces for the compression toolkit's core operations.

These ``Protocol`` classes document the contract each family of components
implements. They are intentionally lightweight (structural typing) so existing
concrete classes satisfy them without inheritance, while new implementations get
a clear, checkable target.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class Pruner(Protocol):
    """Removes weights/structure from a model to reduce size and compute."""

    def apply(self, model: nn.Module, data: Any = None) -> nn.Module:
        """Return a pruned copy (or in-place mutation) of ``model``."""
        ...


@runtime_checkable
class Quantizer(Protocol):
    """Reduces the numeric precision of a model's parameters/activations."""

    def apply(self, model: nn.Module, data: Any = None) -> nn.Module:
        """Return a quantized model."""
        ...


@runtime_checkable
class Distiller(Protocol):
    """Trains a (smaller) student model to mimic a teacher."""

    def apply(self, model: nn.Module, data: Any = None) -> nn.Module:
        """Return the trained student model."""
        ...


@runtime_checkable
class Exporter(Protocol):
    """Serializes a model to a deployable on-disk format."""

    def export(self, model: nn.Module, out_path: str, **kwargs: Any) -> str:
        """Write ``model`` to ``out_path`` and return the path written."""
        ...


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Measures a model's runtime characteristics (latency, memory, size)."""

    def run(self, model: nn.Module, **kwargs: Any) -> dict[str, Any]:
        """Return a dict of measured metrics."""
        ...
