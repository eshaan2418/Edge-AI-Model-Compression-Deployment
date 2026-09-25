"""Pareto frontiers over accuracy x latency x size (x energy once measured), per machine.

Rows come from ``experiments.csv``; latency is the median of per-process medians.
Rows measured on different machines (fingerprints) are never mixed. Energy is
included as an objective only when every row of that machine has it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from edge_ai_compression.optimization.pareto.frontier import ParetoOptimizer


def frontier(exps: pd.DataFrame) -> pd.DataFrame:
    """Frontier rows per fingerprint, with an ``on_frontier`` flag on every row."""
    out = []
    for _fp, machine in exps.groupby("fingerprint_hash"):
        minimize = ["latency_median", "size_mb"]
        if machine["energy_j_per_inf"].notna().all():
            minimize.append("energy_j_per_inf")
        rows = machine.to_dict(orient="records")
        front = ParetoOptimizer(maximize=("accuracy",), minimize=tuple(minimize)).compute_frontier(
            rows
        )
        ids = {r["experiment_id"] for r in front}
        out.append(machine.assign(on_frontier=machine["experiment_id"].isin(ids)))
    return pd.concat(out, ignore_index=True) if out else exps.assign(on_frontier=False)


def plot_frontier(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for backend, g in table.groupby("backend"):
        ax.scatter(g["latency_median"], g["accuracy"], s=14, alpha=0.6, label=backend)
    front = table[table["on_frontier"]].sort_values("latency_median")
    ax.plot(front["latency_median"], front["accuracy"], color="black", lw=1, label="frontier")
    ax.set(xscale="log", xlabel="Latency, median of process medians (ms)", ylabel="Accuracy")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
