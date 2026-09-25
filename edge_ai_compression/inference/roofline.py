"""Roofline analysis over kernel-benchmark rows.

Attainable throughput for a kernel with arithmetic intensity ``AI`` (FLOPs per
byte of compulsory traffic) is ``min(peak_compute, AI * bandwidth)``; kernels
left of the ridge point (``peak / bandwidth``) are memory-bound. Peaks come from
the single-core microbenchmarks (``peak_fma_f32``, ``peak_dot_s8``, ``triad``),
matching the single-threaded kernels.

Caveat: compulsory traffic (each operand moved once) is a lower bound on real
traffic, so AI is an upper bound and kernels may be more memory-bound than the
plot suggests. Efficiency is reported against this optimistic bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MachinePeaks:
    fp32_gflops: float
    s8_gops: float
    bandwidth_gbps: float

    def compute_peak(self, op: str) -> float:
        return self.s8_gops if op == "gemm_s8" else self.fp32_gflops

    def ridge(self, op: str) -> float:
        return self.compute_peak(op) / self.bandwidth_gbps


def peaks_from_rows(df: pd.DataFrame) -> MachinePeaks:
    """Median of logged peak rows (one machine: filter by fingerprint_hash first)."""
    if df["fingerprint_hash"].nunique() > 1:
        raise ValueError("rows span several machines; filter by fingerprint_hash")

    def med(op: str) -> float:
        vals = df.loc[df["op"] == op, "achieved"]
        if vals.empty:
            raise ValueError(f"no '{op}' rows; run the study with peaks enabled")
        return float(vals.median())

    return MachinePeaks(med("peak_fma_f32"), med("peak_dot_s8"), med("triad"))


def roofline_point(op: str, ai: float, achieved: float, peaks: MachinePeaks) -> dict[str, Any]:
    peak = peaks.compute_peak(op)
    bound = min(peak, ai * peaks.bandwidth_gbps)
    return {
        "attainable": bound,
        "efficiency": achieved / bound if bound > 0 else float("nan"),
        "regime": "memory" if ai < peaks.ridge(op) else "compute",
    }


def roofline_table(df: pd.DataFrame, peaks: MachinePeaks) -> pd.DataFrame:
    """Add attainable / efficiency / regime columns to GEMM rows."""
    gemm = df[df["op"].str.startswith("gemm_")].copy()
    points = [
        roofline_point(r.op, float(r.arithmetic_intensity), float(r.achieved), peaks)
        for r in gemm.itertuples()
    ]
    return pd.concat([gemm.reset_index(drop=True), pd.DataFrame(points)], axis=1)


def plot_roofline(table: pd.DataFrame, peaks: MachinePeaks, path: Path) -> Path:
    """Log-log roofline with one marker per op (needs the ``viz`` extra)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ai = np.logspace(-2, 3, 200)
    for op, style in (("gemm_f32", "-"), ("gemm_s8", "--")):
        ax.plot(
            ai,
            np.minimum(peaks.compute_peak(op), ai * peaks.bandwidth_gbps),
            style,
            color="0.4",
            lw=1,
            label=f"roof ({'int8' if op == 'gemm_s8' else 'fp32'})",
        )
    for op, group in table.groupby("op"):
        ax.scatter(group["arithmetic_intensity"], group["achieved"], s=18, label=op)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Arithmetic intensity (FLOP/byte, compulsory)",
        ylabel="Achieved GFLOP/s (GOP/s for int8)",
    )
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
