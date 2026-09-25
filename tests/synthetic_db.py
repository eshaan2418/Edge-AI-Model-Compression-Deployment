"""Synthetic experiment DBs with planted ground truth (DECISIONS D6.2).

Used only to check that the analysis pipelines recover a known relationship.
Nothing here is a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edge_ai_compression.pretraining.signals import SCALAR_FIELDS

W4A8 = ("none", "rtn:w4a8:per_channel")


def write_early_prediction_db(root: Path, seed: int = 0) -> Path:
    """8 configurations x 3 seeds. Final accuracy drop under W4A8 = 0.02 * final
    kurtosis_max + small noise; kurtosis is visible early; other signals are noise."""
    rng = np.random.default_rng(seed)
    runs, signals, exps = [], [], []
    steps = [0, 10, 30, 100, 300, 1000]
    for m, params in enumerate([1e5, 5e5, 2e6, 1e7]):
        for variant in ("standard", "kurtosis"):
            k_config = 3.0 + 4.0 * rng.random()
            for s in range(3):
                run_id = f"m{m}-{variant}-s{s}"
                k_final = k_config + 0.1 * rng.standard_normal()
                runs.append(
                    {
                        "run_id": run_id,
                        "model": f"model{m}",
                        "variant": variant,
                        "variant_options": "{}",
                        "seed": s,
                        "epochs": 10,
                        "steps": 1000,
                    }
                )
                for step in steps:
                    row = {"run_id": run_id, "step": step, "epoch": 0}
                    row.update({k: float(rng.random()) for k in SCALAR_FIELDS})
                    row["kurtosis_max"] = (
                        k_final * (0.6 + 0.4 * step / 1000) + 0.05 * rng.standard_normal()
                    )
                    signals.append(row)
                exps.append(
                    {
                        "source_run_id": run_id,
                        "source_step": 1000,
                        "pruning_type": W4A8[0],
                        "quantization_type": W4A8[1],
                        "accuracy_drop": 0.02 * k_final + 0.002 * rng.standard_normal(),
                        "num_params": int(params),
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(root / "training_runs.csv", index=False)
    pd.DataFrame(signals).to_csv(root / "training_signals.csv", index=False)
    pd.DataFrame(exps).to_csv(root / "experiments.csv", index=False)
    return root


def write_scaling_db(root: Path, seed: int = 0) -> Path:
    """Planted power law for `standard` (b = 0.4) and a flat series for `flat`."""
    rng = np.random.default_rng(seed)
    runs, exps = [], []
    for params in (1e5, 3e5, 1e6, 3e6, 1e7):
        for variant in ("standard", "flat"):
            for s in range(3):
                run_id = f"{variant}-{int(params)}-s{s}"
                runs.append(
                    {
                        "run_id": run_id,
                        "model": f"m{int(params)}",
                        "variant": variant,
                        "variant_options": "{}",
                        "seed": s,
                        "epochs": 30,
                        "steps": 100,
                    }
                )
                drop = 0.3 * (params / 1e5) ** -0.4 + 0.02 if variant == "standard" else 0.05
                exps.append(
                    {
                        "source_run_id": run_id,
                        "source_step": 100,
                        "pruning_type": W4A8[0],
                        "quantization_type": W4A8[1],
                        "accuracy_drop": drop + 0.003 * rng.standard_normal(),
                        "num_params": int(params),
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(root / "training_runs.csv", index=False)
    pd.DataFrame(exps).to_csv(root / "experiments.csv", index=False)
    return root
