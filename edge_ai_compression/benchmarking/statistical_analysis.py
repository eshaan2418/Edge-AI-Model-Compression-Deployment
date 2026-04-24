from __future__ import annotations

import numpy as np

from edge_ai_compression.utils.metrics import compute_latency_stats


def summarize_latencies(latencies_ms: list[float] | np.ndarray) -> dict[str, float]:
    return compute_latency_stats(latencies_ms)
