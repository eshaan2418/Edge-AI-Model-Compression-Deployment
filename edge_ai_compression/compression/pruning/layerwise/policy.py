from __future__ import annotations

import math


def scores_to_sparsities(
    scores: dict[str, float],
    mean_sparsity: float,
    *,
    min_sparsity: float = 0.02,
    max_sparsity: float = 0.92,
) -> dict[str, float]:
    """Lower score → higher prune (sparsity)."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if math.fabs(hi - lo) < 1e-12:
        return {k: mean_sparsity for k in scores}
    norm = {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    raw = {k: mean_sparsity * (1.0 + (1.0 - nv)) for k, nv in norm.items()}
    mean_raw = sum(raw.values()) / len(raw)
    scale = mean_sparsity / max(mean_raw, 1e-6)
    out = {k: float(min(max_sparsity, max(min_sparsity, v * scale))) for k, v in raw.items()}
    return out
