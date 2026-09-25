"""Analysis of kernel-study rows (``kernel_benchmarks.csv``).

``sparsity_table``: for every GEMM shape, each sparse/quantized kernel's median
latency relative to dense fp32 on the same shape, ISA and machine, with a CI on
the ratio. ``crossover``: the lowest logged sparsity at which the unstructured
CSR kernel beats dense (the CI upper bound < 1, i.e. a significant win).

Only rows from one machine (fingerprint_hash) and one study may be mixed.
The latency ratio CI is a bootstrap over per-process medians
(``benchmarking.stats.compare``), read from the JSONL detail file.

Usage:
    python -m edge_ai_compression.analysis.kernel_study --results results --study kernel_sparsity_m5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from edge_ai_compression.benchmarking.stats import compare
from edge_ai_compression.experiment_db.paths import kernel_benchmarks_jsonl

SHAPE = ["m", "n", "k"]


def load_rows(results_dir: Path, study: str) -> pd.DataFrame:
    rows = [
        json.loads(line) for line in kernel_benchmarks_jsonl(results_dir).read_text().splitlines()
    ]
    df = pd.DataFrame(rows)
    df = df[df["study"] == study]
    if df.empty:
        raise ValueError(f"no rows for study '{study}' in {results_dir}")
    if df["fingerprint_hash"].nunique() != 1:
        raise ValueError("study rows span several machines; analyze one fingerprint at a time")
    return df.reset_index(drop=True)


def sparsity_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (shape, isa, op, sparsity): ratio of median latency vs dense fp32."""
    gemm = df[df["op"].str.startswith("gemm_")]
    out = []
    for (m, n, k, isa), group in gemm.groupby([*SHAPE, "isa"]):
        dense = group[group["op"] == "gemm_f32"]
        if dense.empty:
            continue
        base = dense.iloc[0]
        for r in group.itertuples():
            if r.op == "gemm_f32":
                continue
            row = {
                "m": m,
                "n": n,
                "k": k,
                "isa": isa,
                "op": r.op,
                "sparsity": r.sparsity,
                "latency_median_us": r.latency_median_us,
                "dense_latency_median_us": base["latency_median_us"],
            }
            if len(base["process_medians_us"]) > 1 and len(r.process_medians_us) > 1:
                c = compare(base["process_medians_us"], r.process_medians_us)
                row.update(
                    ratio=c.ratio,
                    ratio_ci_lo=c.ratio_ci[0],
                    ratio_ci_hi=c.ratio_ci[1],
                    p_value=c.p_value,
                )
            else:
                row.update(
                    ratio=r.latency_median_us / base["latency_median_us"],
                    ratio_ci_lo=float("nan"),
                    ratio_ci_hi=float("nan"),
                    p_value=float("nan"),
                )
            out.append(row)
    return pd.DataFrame(out)


def crossover(table: pd.DataFrame) -> pd.DataFrame:
    """Per shape: lowest CSR sparsity with ratio CI entirely below 1 (NaN if none)."""
    csr = table[table["op"] == "gemm_csr"].sort_values("sparsity")
    rows = []
    for (m, n, k, isa), group in csr.groupby([*SHAPE, "isa"]):
        wins = group[group["ratio_ci_hi"] < 1.0]
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "isa": isa,
                "crossover_sparsity": float(wins["sparsity"].min()) if len(wins) else float("nan"),
                "max_sparsity_tested": float(group["sparsity"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_sparsity(table: pd.DataFrame, path: Path) -> Path:
    """Latency ratio vs sparsity per shape (CSR lines, 2:4 markers at 0.5)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, n, k), group in table.groupby(SHAPE):
        csr = group[group["op"] == "gemm_csr"].sort_values("sparsity")
        line = ax.plot(csr["sparsity"], csr["ratio"], marker="o", ms=3, lw=1, label=f"{m}x{n}x{k}")[
            0
        ]
        s24 = group[group["op"] == "gemm_sparse24"]
        if len(s24):
            ax.scatter([0.5], s24["ratio"], marker="*", s=60, color=line.get_color())
    ax.axhline(1.0, color="0.3", lw=0.8, ls="--")
    ax.set(xlabel="Weight sparsity", ylabel="Latency / dense fp32 latency", yscale="log")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--study", required=True)
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    args = p.parse_args()
    table = sparsity_table(load_rows(args.results, args.study))
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"{args.study}_sparsity_table.csv", index=False)
    crossover(table).to_csv(args.out / f"{args.study}_crossover.csv", index=False)
    print(plot_sparsity(table, args.out / f"{args.study}_sparsity.png"))


if __name__ == "__main__":
    main()
