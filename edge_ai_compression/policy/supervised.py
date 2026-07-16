from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def suggest_config_from_table(
    csv_path: str | Path, constraints: dict[str, float]
) -> dict[str, Any]:
    """Pick the best row under simple scalarized score (Phase 5 supervised baseline)."""
    df = pd.read_csv(csv_path)
    max_lat = constraints.get("max_latency_ms", 1e9)
    max_size = constraints.get("max_size_mb", 1e9)
    min_acc = constraints.get("min_accuracy", 0.0)
    sub = df[
        (df["latency_mean"] <= max_lat) & (df["size_mb"] <= max_size) & (df["accuracy"] >= min_acc)
    ]
    if sub.empty:
        sub = df
    score = sub["accuracy"] - 0.01 * sub["latency_mean"] - 0.02 * sub["size_mb"]
    best = sub.iloc[int(score.argmax())]
    return {
        "compression_order": best.get("compression_order", ""),
        "pruning_sparsity": float(best.get("pruning_sparsity", 0)),
        "row": best.to_dict(),
    }
