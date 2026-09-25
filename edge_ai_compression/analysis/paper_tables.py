"""LaTeX tables for the paper, generated from the experiment DB and reproduce.sh outputs.

Every table is written to ``paper/generated/<name>.tex``. When its inputs do not
exist yet the file contains ``\\pending{<what to run>}`` instead, so the paper
compiles at every stage and never contains a number that did not come from the DB.
Aggregates over seeds are reported as mean [95% percentile-bootstrap CI] with n.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from edge_ai_compression.analysis.common import read_table
from edge_ai_compression.benchmarking.stats import bootstrap_ci

LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
}


def escape(text: object) -> str:
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in str(text))


def fmt(x: float, digits: int = 3) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{digits}g}"


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(lo):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} [{fmt(lo, digits)}, {fmt(hi, digits)}]"


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    v = values.dropna().to_numpy(float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size < 2:
        return float(v.mean()), float("nan"), float("nan")
    lo, hi = bootstrap_ci(v, np.mean, n_boot=2000)
    return float(v.mean()), lo, hi


def tabular(header: list[str], rows: list[list[str]], caption: str) -> str:
    cols = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", f"% {caption}"]
    return "\n".join(lines) + "\n"


def pending(what: str) -> str:
    return "\\pending{" + escape(what) + "}\n"


LADDER_COLUMNS = {
    "model_name",
    "dataset",
    "backend",
    "pruning_type",
    "quantization_type",
    "accuracy",
    "accuracy_drop",
    "latency_median",
    "size_mb",
    "weight_sparsity",
}


def _ladder(exps: pd.DataFrame, key: str, backends: set[str], other: str) -> str | None:
    if not set(exps.columns) >= LADDER_COLUMNS:
        return None
    df = exps[
        (exps["model_name"] == "resnet18_cifar")
        & (exps["dataset"] == "cifar10")
        & exps["backend"].isin(backends)
        & (exps[other] == "none")
    ]
    if df.empty:
        return None
    rows = []
    for tag, g in df.groupby(key, sort=False):
        acc, drop, lat = (
            mean_ci(g["accuracy"]),
            mean_ci(g["accuracy_drop"]),
            mean_ci(g["latency_median"]),
        )
        row = [
            escape(tag),
            fmt_ci(*acc),
            fmt_ci(*drop),
            fmt_ci(*lat),
            fmt(g["size_mb"].mean()),
            str(len(g)),
        ]
        if key == "pruning_type":
            row.insert(4, fmt(g["weight_sparsity"].mean()))
        rows.append(row)
    header = ["Rung", "Accuracy", "Acc. drop", "Latency (ms)", "Size (MB)", "n"]
    if key == "pruning_type":
        header.insert(4, "Sparsity")
    return tabular(header, rows, f"ResNet-18 / CIFAR-10 ladder grouped by {key}")


def ladder_ptq(results: Path, figures: Path) -> str:
    todo = pending("run configs/sweeps/ptq_ladder_resnet18_cifar10.yml")
    if not (results / "experiments.csv").is_file():
        return todo
    exps = read_table(results / "experiments.csv")
    return _ladder(exps, "quantization_type", {"edge_quant", "edge_f32"}, "pruning_type") or todo


def ladder_prune(results: Path, figures: Path) -> str:
    todo = pending("run configs/sweeps/prune_ladder_resnet18_cifar10.yml")
    if not (results / "experiments.csv").is_file():
        return todo
    exps = read_table(results / "experiments.csv")
    backends = {"edge_csr", "edge_sparse24", "edge_f32"}
    return _ladder(exps, "pruning_type", backends, "quantization_type") or todo


def early_prediction(results: Path, figures: Path, method: str = "w4a8") -> str:
    path = figures / f"early_prediction_{method}.csv"
    if not path.is_file():
        return pending(f"run the Phase 5 tracks + compress_runs, then reproduce.sh ({method})")
    t = pd.read_csv(path)
    fractions = sorted(t["fraction"].unique())
    rows = []
    for name, g in t.groupby("predictor", sort=False):
        g = g.set_index("fraction")
        rows.append(
            [escape(name)]
            + [
                fmt_ci(g.loc[f, "spearman"], g.loc[f, "spearman_lo"], g.loc[f, "spearman_hi"], 2)
                for f in fractions
            ]
        )
    header = ["Predictor"] + [f"f={f:g}" for f in fractions]
    return tabular(header, rows, f"Spearman rho, leave-one-configuration-out, method {method}")


def scaling(results: Path, figures: Path) -> str:
    paths = sorted(figures.glob("scaling_params_*.csv"))
    if not paths:
        return pending("run pretrain_scaling + compress_runs, then reproduce.sh")
    rows = []
    for p in paths:
        for r in pd.read_csv(p).itertuples():
            rows.append(
                [
                    escape(r.method),
                    escape(r.variant),
                    fmt_ci(r.b, r.b_lo, r.b_hi, 2),
                    "yes" if r.beats_constant else "no",
                    str(r.n_points),
                ]
            )
    return tabular(
        ["Method", "Variant", "Exponent b", "Beats const.", "Sizes"],
        rows,
        "drop = a (N/N0)^-b + c over parameter count",
    )


def latency_proxy(results: Path, figures: Path) -> str:
    paths = sorted(figures.glob("latency_proxy_*.csv"))
    if not paths:
        return pending(
            "run the kernel study and backend_latency_m5 on the M5 Pro, then reproduce.sh"
        )
    rows = []
    for p in paths:
        for r in pd.read_csv(p).itertuples():
            rows.append([escape(r.proxy), fmt(r.spearman, 3), fmt(r.mape, 3), str(r.n)])
    return tabular(
        ["Proxy", "Spearman", "MAPE", "n"],
        rows,
        "latency proxies, leave-one-configuration-out calibration",
    )


def kernel_crossover(results: Path, figures: Path) -> str:
    paths = sorted(figures.glob("*_crossover.csv"))
    if not paths:
        return pending("run configs/studies/kernel_sparsity_m5.yml, then reproduce.sh")
    rows = []
    for p in paths:
        for r in pd.read_csv(p).itertuples():
            rows.append(
                [
                    f"{r.m}x{r.n}x{r.k}",
                    escape(r.isa),
                    fmt(r.crossover_sparsity, 2),
                    fmt(r.max_sparsity_tested, 2),
                ]
            )
    return tabular(
        ["GEMM (MxNxK)", "ISA", "CSR crossover sparsity", "Max tested"],
        rows,
        "lowest sparsity where CSR beats dense fp32 (ratio CI < 1)",
    )


def lm_degradation(results: Path, figures: Path) -> str:
    path = figures / "lm_degradation.csv"
    if not path.is_file():
        return pending("run configs/lm/tinystories_gpt_*.yml (3 seeds), then reproduce.sh")
    t = pd.read_csv(path)
    t = t[t["stage"] != "pretrain"]
    rows = [
        [
            escape(r.model),
            escape(r.stage),
            escape(r.method),
            fmt_ci(r.rel_capability, r.rel_capability_lo, r.rel_capability_hi, 3),
            fmt_ci(r.rel_behavior, r.rel_behavior_lo, r.rel_behavior_hi, 3),
            fmt_ci(r.gap, r.gap_lo, r.gap_hi, 2),
            str(r.n_seeds),
        ]
        for r in t.itertuples()
    ]
    return tabular(
        ["Model", "Stage", "Method", "Capability", "Behavior", "Gap", "Seeds"],
        rows,
        "relative to float; gap = capability - behavior",
    )


TABLES: dict[str, Callable[[Path, Path], str]] = {
    "ladder_ptq": ladder_ptq,
    "ladder_prune": ladder_prune,
    "early_prediction_w4a8": early_prediction,
    "scaling_params": scaling,
    "latency_proxy": latency_proxy,
    "kernel_crossover": kernel_crossover,
    "lm_degradation": lm_degradation,
}


def write_paper_tables(results: Path, figures: Path, out: Path) -> dict[str, bool]:
    """Write every table; returns {name: True if real data, False if pending}."""
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, fn in TABLES.items():
        try:
            text = fn(results, figures)
        except FileNotFoundError:
            text = pending("run the experiments that produce experiments.csv")
        (out / f"{name}.tex").write_text(text)
        status[name] = not text.startswith("\\pending")
    return status
