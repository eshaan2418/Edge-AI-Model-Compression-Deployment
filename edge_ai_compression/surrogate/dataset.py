from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    y_acc = df["accuracy"].to_numpy(dtype=np.float64)
    y_lat = df["latency_mean"].to_numpy(dtype=np.float64)
    y_size = df["size_mb"].to_numpy(dtype=np.float64)
    y_ram = df["ram_mb"].to_numpy(dtype=np.float64)
    y_energy = df["energy_proxy"].to_numpy(dtype=np.float64)
    return X, {
        "accuracy": y_acc,
        "latency_mean": y_lat,
        "size_mb": y_size,
        "ram_mb": y_ram,
        "energy_proxy": y_energy,
    }
