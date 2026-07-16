from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FailureAnalyzer:
    def compare(
        self, baseline: nn.Module, compressed: nn.Module, loader: DataLoader, device: str
    ) -> dict[str, Any]:
        baseline.eval()
        compressed.eval()
        mismatches = 0
        total = 0
        baseline_right_compressed_wrong = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                b_pred = baseline(x).argmax(dim=1)
                c_pred = compressed(x).argmax(dim=1)
                total += y.numel()
                mismatches += (c_pred != y).sum().item()
                baseline_right_compressed_wrong += ((b_pred == y) & (c_pred != y)).sum().item()
        return {
            "compressed_error_rate": mismatches / max(total, 1),
            "regression_rate_vs_baseline": baseline_right_compressed_wrong / max(total, 1),
        }
