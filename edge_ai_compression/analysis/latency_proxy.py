"""Do FLOPs, parameters, or sparsity predict measured latency? And a learned proxy that does.

Part 1 (naive proxies): Spearman correlation of FLOPs, parameter count, measured
weight sparsity and artifact size with measured latency (median of process
medians), per (machine, backend) and pooled across backends on one machine.
Pooling is where FLOPs is expected to mislead: kernels differ in achieved
FLOP/s (dense int8 vs sparse CSR), so equal FLOPs do not mean equal time.

Part 2 (learned proxy): a per-layer kernel-latency model fit on
``kernel_benchmarks`` rows of the same machine (ridge regression on log shape
features per kernel op, which extrapolates smoothly across shapes), summed over
a model's recorded layer GEMM shapes, then calibrated per run configuration
held out (measured = alpha * sum + beta, fit on the *other* configurations, to
absorb non-GEMM overhead such as im2col, requantization and residual adds).
Compared against FLOPs-only and params-only linear proxies evaluated the same way.

    python -m edge_ai_compression.analysis.latency_proxy --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

from edge_ai_compression.analysis.common import InsufficientData, read_table
from edge_ai_compression.experiment_db.paths import artifact_dir

BACKEND_OP = {
    "edge_f32": "gemm_f32",
    "edge_int8": "gemm_s8",
    "edge_quant": "gemm_s8",
    "edge_w4": "gemm_w4",
    "edge_sparse24": "gemm_sparse24",
    "edge_csr": "gemm_csr",
}
NAIVE = ("flops", "num_params", "weight_sparsity", "size_mb")


def _rho(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def naive_correlations(exps: pd.DataFrame) -> pd.DataFrame:
    """Spearman(proxy, latency_median) per (fingerprint, backend) and pooled per fingerprint."""
    rows = []
    for fp, machine in exps.groupby("fingerprint_hash"):
        groups = [(b, g) for b, g in machine.groupby("backend")] + [("all backends", machine)]
        for backend, g in groups:
            row = {"fingerprint_hash": fp, "backend": backend, "n": len(g)}
            row.update({f"rho_{p}": _rho(g[p], g["latency_median"]) for p in NAIVE if p in g})
            rows.append(row)
    return pd.DataFrame(rows)


def _features(m: np.ndarray, n: np.ndarray, k: np.ndarray, sparsity: np.ndarray) -> np.ndarray:
    lm, ln, lk = np.log(m), np.log(n), np.log(k)
    return np.column_stack([lm, ln, lk, lm + ln + lk, np.log1p(-np.minimum(sparsity, 0.999))])


class KernelLatencyModel:
    """Per-op ridge regression of log latency (us) on log GEMM shape features."""

    def __init__(self) -> None:
        self.models: dict[str, Ridge] = {}

    def fit(self, kernels: pd.DataFrame) -> KernelLatencyModel:
        gemm = kernels[kernels["op"].str.startswith("gemm_") & kernels["latency_median_us"].notna()]
        for op, g in gemm.groupby("op"):
            x = _features(
                g["m"].to_numpy(float),
                g["n"].to_numpy(float),
                g["k"].to_numpy(float),
                g["sparsity"].to_numpy(float),
            )
            self.models[op] = Ridge(alpha=1e-3).fit(
                x, np.log(g["latency_median_us"].to_numpy(float))
            )
        return self

    def predict_us(self, op: str, m: float, n: float, k: float, sparsity: float) -> float:
        if op not in self.models:
            raise KeyError(f"no kernel benchmarks for {op}")
        x = _features(np.array([m]), np.array([n]), np.array([k]), np.array([sparsity]))
        return float(np.exp(self.models[op].predict(x)[0]))

    def model_latency_us(self, backend: str, shapes: list[dict], sparsity: float) -> float:
        op = BACKEND_OP[backend]
        return sum(
            self.predict_us(
                op,
                s["m"],
                s["n"],
                s["k"],
                sparsity if op == "gemm_csr" else (0.5 if op == "gemm_sparse24" else 0.0),
            )
            for s in shapes
        )


def load_layer_shapes(results_dir: Path, experiment_id: str) -> list[dict]:
    metrics = json.loads((artifact_dir(results_dir, experiment_id) / "metrics.json").read_text())
    return metrics["layer_shapes"]


def _loco(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-configuration-out predictions of a 1-D linear calibration y ~ a x + b."""
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        model = LinearRegression().fit(x[~test].reshape(-1, 1), y[~test])
        pred[test] = model.predict(x[test].reshape(-1, 1))
    return pred


def _quality(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, pred).statistic),
        "mape": float(np.mean(np.abs(pred - y) / y)),
    }


def proxy_comparison(
    exps: pd.DataFrame, kernel_model: KernelLatencyModel, shapes_by_id: dict[str, list[dict]]
) -> pd.DataFrame:
    """Learned vs FLOPs vs params proxies on one machine's engine-backend runs."""
    df = exps[exps["backend"].isin(BACKEND_OP) & exps["experiment_id"].isin(shapes_by_id)].copy()
    df["learned_sum_ms"] = [
        kernel_model.model_latency_us(r.backend, shapes_by_id[r.experiment_id], r.weight_sparsity)
        / 1e3
        for r in df.itertuples()
    ]
    groups = (
        df["model_name"]
        + "|"
        + df["pruning_type"]
        + "|"
        + df["quantization_type"]
        + "|"
        + df["backend"]
    ).to_numpy()
    if len(np.unique(groups)) < 3:
        raise InsufficientData("need >= 3 run configurations for leave-one-configuration-out")
    y = df["latency_median"].to_numpy(float)
    rows = []
    for name, col in (("learned", "learned_sum_ms"), ("flops", "flops"), ("params", "num_params")):
        pred = _loco(df[col].to_numpy(float), y, groups)
        rows.append({"proxy": name, "n": len(df), **_quality(y, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    exps = read_table(a.results / "experiments.csv")
    naive_correlations(exps).to_csv(a.out / "latency_naive_correlations.csv", index=False)
    kernels = read_table(a.results / "kernel_benchmarks.csv")
    for fp, k in kernels.groupby("fingerprint_hash"):
        machine = exps[exps["fingerprint_hash"] == fp]
        shapes = {}
        for eid in machine["experiment_id"]:
            try:
                shapes[eid] = load_layer_shapes(a.results, eid)
            except (FileNotFoundError, KeyError):
                continue
        table = proxy_comparison(machine, KernelLatencyModel().fit(k), shapes)
        table.to_csv(a.out / f"latency_proxy_{fp}.csv", index=False)
        print(table)


if __name__ == "__main__":
    main()
