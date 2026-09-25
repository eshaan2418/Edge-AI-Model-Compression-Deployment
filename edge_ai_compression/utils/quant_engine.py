"""Select a PyTorch quantized backend engine.

Some builds (e.g. torch 2.14 on macOS arm64) start with engine ``none``, and any
quantized op, including unpickling a dynamically quantized module, then fails
with ``NoQEngine``.
"""

from __future__ import annotations

import torch

PREFERRED_ENGINES = ("x86", "fbgemm", "qnnpack")


def ensure_quantized_engine() -> str:
    """Set the quantized engine if none is selected; return the engine in use."""
    qb = torch.backends.quantized
    if qb.engine == "none":
        available = [e for e in PREFERRED_ENGINES if e in qb.supported_engines]
        if not available:
            raise RuntimeError(f"no quantized engine available; supported: {qb.supported_engines}")
        qb.engine = available[0]
    return qb.engine
