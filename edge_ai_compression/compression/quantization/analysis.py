from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def weight_histogram_stats(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu().numpy().ravel()
    return {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "sqnr_db_estimate": float(10 * np.log10((np.mean(w**2) + 1e-12) / (np.var(w) * 1e-6 + 1e-12))),
    }


def layer_quantization_report(model: nn.Module) -> list[dict[str, Any]]:
    """Phase 10 — per-layer weight statistics (pre/post compare in caller)."""
    rows: list[dict[str, Any]] = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            rows.append({"layer": name, **weight_histogram_stats(m.weight)})
    return rows
