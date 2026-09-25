from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Energy is not a target until a real meter fills energy_j_per_inf (DECISIONS D1.9).
TARGETS: tuple[str, ...] = ("accuracy", "latency_median", "size_mb", "peak_rss_mib")


def load_experiment_table(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Feature matrix X and dict of target vectors for multi-output training."""
    feature_cols = [
        "num_params",
        "flops",
        "pruning_sparsity",
        "temperature",
        "alpha",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return X, {k: df[k].to_numpy(dtype=np.float64) for k in TARGETS}
