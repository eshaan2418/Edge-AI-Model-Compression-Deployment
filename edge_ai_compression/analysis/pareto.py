"""CLI: compute the Pareto frontier from an experiment results table.

Reads a results CSV (default ``results/experiments.csv``) and reports the set of
non-dominated compression configurations across accuracy (maximize) and resource
costs (minimize). Writes ``pareto_frontier.csv`` and ``pareto_frontier.md``, and
optionally a scatter plot.

    python -m edge_ai_compression.analysis.pareto \
        --results results/experiments.csv \
        --out results/pareto \
        --plot

Only objectives that are actually present (and non-null) in the table are used,
so partially-populated result sets are handled without crashing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from edge_ai_compression.optimization.pareto.frontier import ParetoOptimizer

# Canonical objective columns and their optimization direction.
_MAXIMIZE = ("accuracy",)
_MINIMIZE = ("latency_mean", "size_mb", "ram_mb")

# Columns carried through to the report for context (if present).
_LABEL_COLUMNS = (
    "experiment_id",
    "model_name",
    "dataset",
    "compression_order",
    "pruning_sparsity",
    "quantization_type",
)


def _select_objectives(
    columns: list[str],
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep only objectives that exist as columns."""
    present = set(columns)
    return (
        tuple(m for m in maximize if m in present),
        tuple(m for m in minimize if m in present),
    )


def compute_pareto(
    df: pd.DataFrame,
    maximize: tuple[str, ...] = _MAXIMIZE,
    minimize: tuple[str, ...] = _MINIMIZE,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Return the Pareto-optimal rows of ``df`` and the objectives used.

    Rows missing any used objective value are dropped before comparison so a
    single incomplete row cannot corrupt the frontier.
    """
    max_obj, min_obj = _select_objectives(list(df.columns), maximize, minimize)
    used = list(max_obj) + list(min_obj)
    if not used:
        raise ValueError(
            "No known objective columns found. Expected some of: "
            f"{list(maximize) + list(minimize)}. Got columns: {list(df.columns)}."
        )

    clean = df.dropna(subset=used).reset_index(drop=True)
    if clean.empty:
        return clean, max_obj, min_obj

    rows: list[dict[str, Any]] = clean.to_dict(orient="records")
    optimizer = ParetoOptimizer(maximize=max_obj, minimize=min_obj)
    frontier = optimizer.compute_frontier(rows)

    frontier_df = pd.DataFrame(frontier)
    # Preserve column order from the source table.
    ordered_cols = [c for c in df.columns if c in frontier_df.columns]
    return frontier_df[ordered_cols], max_obj, min_obj


def _to_markdown(
    frontier: pd.DataFrame,
    max_obj: tuple[str, ...],
    min_obj: tuple[str, ...],
    total: int,
) -> str:
    display_cols = [c for c in (*_LABEL_COLUMNS, *max_obj, *min_obj) if c in frontier.columns]
    lines = [
        "# Pareto Frontier",
        "",
        f"- Non-dominated configurations: **{len(frontier)}** of {total} evaluated",
        f"- Maximize: {', '.join(max_obj) or '(none)'}",
        f"- Minimize: {', '.join(min_obj) or '(none)'}",
        "",
    ]
    if frontier.empty or not display_cols:
        lines.append("_No frontier points to display._")
        return "\n".join(lines) + "\n"

    view = frontier[display_cols]
    lines.append("| " + " | ".join(display_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in display_cols) + " |")
    for _, row in view.iterrows():
        cells = []
        for col in display_cols:
            val = row[col]
            cells.append(f"{val:.4g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _maybe_plot(
    frontier: pd.DataFrame,
    all_df: pd.DataFrame,
    max_obj: tuple[str, ...],
    min_obj: tuple[str, ...],
    out_path: Path,
) -> bool:
    """Draw accuracy-vs-cost scatter. Returns True on success, False if skipped."""
    if not max_obj or not min_obj:
        print("Plot skipped: need at least one maximize and one minimize objective.")
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: matplotlib is not installed (pip install matplotlib).")
        return False

    y_col = max_obj[0]
    x_col = min_obj[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_df[x_col], all_df[y_col], c="lightgray", label="dominated", zorder=1)
    front_sorted = frontier.sort_values(x_col)
    ax.plot(
        front_sorted[x_col],
        front_sorted[y_col],
        "-o",
        color="tab:red",
        label="Pareto frontier",
        zorder=2,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Accuracy vs. cost trade-off")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results",
        type=Path,
        default=Path("results/experiments.csv"),
        help="Path to the experiments CSV.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/pareto"),
        help="Output directory (frontier CSV/MD, and plot if --plot).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also write a scatter plot (needs matplotlib).",
    )
    args = p.parse_args()

    if not args.results.is_file():
        raise SystemExit(
            f"No results at {args.results}. Run experiments with experiment_db enabled first."
        )

    df = pd.read_csv(args.results)
    if df.empty:
        raise SystemExit(f"{args.results} has no rows.")

    frontier, max_obj, min_obj = compute_pareto(df)
    used = list(max_obj) + list(min_obj)
    clean_total = len(df.dropna(subset=used)) if used else len(df)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pareto_frontier.csv"
    md_path = out_dir / "pareto_frontier.md"

    frontier.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(frontier, max_obj, min_obj, clean_total), encoding="utf-8")

    print(f"Objectives  : max={list(max_obj)} min={list(min_obj)}")
    print(f"Frontier    : {len(frontier)} of {clean_total} configurations")
    print(f"Wrote       : {csv_path}")
    print(f"Wrote       : {md_path}")

    if args.plot:
        plot_path = out_dir / "pareto_frontier.png"
        if _maybe_plot(frontier, df.dropna(subset=used), max_obj, min_obj, plot_path):
            print(f"Wrote       : {plot_path}")


if __name__ == "__main__":
    main()
