from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from edge_ai_compression.surrogate.dataset import build_xy, load_experiment_table


def train_surrogates(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    model_types: tuple[str, ...] = ("rf", "mlp", "gp"),
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_experiment_table(csv_path)
    X, ys = build_xy(df)
    Y = np.column_stack(
        [ys[k] for k in ("accuracy", "latency_mean", "size_mb", "ram_mb", "energy_proxy")]
    )
    meta: dict[str, Any] = {"n_rows": len(df), "targets": list(ys.keys())}

    if "rf" in model_types:
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=12, random_state=0, n_jobs=-1)
        )
        rf.fit(X, Y)
        joblib.dump(rf, out_dir / "surrogate_rf.joblib")

    if "mlp" in model_types:
        mlp = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0),
                ),
            ]
        )
        mlp.fit(X, Y)
        joblib.dump(mlp, out_dir / "surrogate_mlp.joblib")

    if "gp" in model_types:
        gp = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gp", GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, random_state=0)),
            ]
        )
        gp.fit(X, Y[:, 0:1])
        joblib.dump(gp, out_dir / "surrogate_gp_accuracy.joblib")

    with open(out_dir / "surrogate_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def predict_rf(out_dir: str | Path, X: np.ndarray) -> np.ndarray:
    m = joblib.load(Path(out_dir) / "surrogate_rf.joblib")
    return m.predict(X)
