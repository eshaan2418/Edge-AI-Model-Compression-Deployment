"""Regenerate every figure and table from the experiment DB (``reproduce.sh``).

Each analysis runs only if its input tables exist; missing inputs are listed as
PENDING in ``results/figures/MANIFEST.md`` (with the config that produces them)
instead of failing, so the manifest always states exactly which results exist.
No number reaches the docs or the paper except through these outputs.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from edge_ai_compression.analysis.errors import InsufficientData
from edge_ai_compression.experiments.compress_runs import PANEL


def _kernel_study(results: Path, out: Path) -> list[Path]:
    from edge_ai_compression.analysis.kernel_study import load_rows, plot_sparsity, sparsity_table
    from edge_ai_compression.inference.roofline import (
        peaks_from_rows,
        plot_roofline,
        roofline_table,
    )

    kernels = pd.read_csv(results / "kernel_benchmarks.csv")
    written = []
    for study in sorted(kernels["study"].unique()):
        if study.startswith("smoke"):
            continue
        rows = load_rows(results, study)
        table = sparsity_table(rows)
        table.to_csv(out / f"{study}_sparsity_table.csv", index=False)
        written.append(plot_sparsity(table, out / f"{study}_sparsity.png"))
        study_rows = kernels[kernels["study"] == study]
        peaks = peaks_from_rows(study_rows)
        roof = roofline_table(study_rows, peaks)
        roof.to_csv(out / f"{study}_roofline.csv", index=False)
        written.append(plot_roofline(roof, peaks, out / f"{study}_roofline.png"))
    return written


def _early_prediction(results: Path, out: Path) -> list[Path]:
    from edge_ai_compression.analysis import early_prediction as ep

    written = []
    runs, signals, exps = ep.load_tables(results)
    for method in PANEL:
        try:
            table = ep.prediction_vs_fraction(results, method)
        except InsufficientData:
            continue  # no (or too few) runs compressed with this method yet
        table.to_csv(out / f"early_prediction_{method}.csv", index=False)
        written.append(ep.plot_vs_fraction(table, out / f"early_prediction_{method}.png"))
        ep.ablation(ep.build_dataset(runs, signals, exps, method, 0.1)).to_csv(
            out / f"ablation_{method}_f0.1.csv", index=False
        )
    return written


def _scaling(results: Path, out: Path) -> list[Path]:
    from edge_ai_compression.analysis import scaling

    written = []
    studies = {
        "params": ("num_params", {"epochs": 30}, "Parameters"),
        "epochs": ("epochs", {"model": "resnet18_w0.5_cifar", "variant": "standard"}, "Epochs"),
    }
    for method in PANEL:
        for name, (x, filters, label) in studies.items():
            table = scaling.scaling_table(results, method, x, filters)
            if table.empty:
                continue
            table.to_csv(out / f"scaling_{name}_{method}.csv", index=False)
            pts = scaling.scaling_points(results, method, x, filters)
            written.append(
                scaling.plot_scaling(pts, table, out / f"scaling_{name}_{method}.png", label)
            )
    return written


def _latency_proxy(results: Path, out: Path) -> list[Path]:
    from edge_ai_compression.analysis import latency_proxy as lp

    exps = pd.read_csv(results / "experiments.csv")
    path = out / "latency_naive_correlations.csv"
    lp.naive_correlations(exps).to_csv(path, index=False)
    written = [path]
    kernels_path = results / "kernel_benchmarks.csv"
    if kernels_path.is_file():
        kernels = pd.read_csv(kernels_path)
        for fp, k in kernels.groupby("fingerprint_hash"):
            machine = exps[exps["fingerprint_hash"] == fp]
            shapes = {}
            for eid in machine["experiment_id"]:
                try:
                    shapes[eid] = lp.load_layer_shapes(results, eid)
                except (FileNotFoundError, KeyError):
                    continue
            try:
                table = lp.proxy_comparison(machine, lp.KernelLatencyModel().fit(k), shapes)
            except InsufficientData:
                continue  # fewer than 3 engine configurations measured on this machine
            p = out / f"latency_proxy_{fp}.csv"
            table.to_csv(p, index=False)
            written.append(p)
    return written


def _pareto(results: Path, out: Path) -> list[Path]:
    from edge_ai_compression.analysis.pareto_report import frontier, plot_frontier

    table = frontier(pd.read_csv(results / "experiments.csv"))
    table.to_csv(out / "pareto.csv", index=False)
    return [plot_frontier(table, out / "pareto.png")]


# name -> (required tables, producing config/notebook, function)
ANALYSES: dict[str, tuple[tuple[str, ...], str, Callable[[Path, Path], list[Path]]]] = {
    "kernel study + roofline": (
        ("kernel_benchmarks.csv", "kernel_benchmarks.jsonl"),
        "configs/studies/kernel_sparsity_m5.yml",
        _kernel_study,
    ),
    "early predictability": (
        ("training_runs.csv", "training_signals.csv", "experiments.csv"),
        "notebooks/tracks.ipynb (pretrain_*) + compress_runs",
        _early_prediction,
    ),
    "scaling curves": (
        ("training_runs.csv", "experiments.csv"),
        "pretrain_scaling / pretrain_length + compress_runs",
        _scaling,
    ),
    "latency proxy": (("experiments.csv",), "backend_latency_m5 + kernel study", _latency_proxy),
    "pareto frontiers": (("experiments.csv",), "any experiment sweep", _pareto),
}


def reproduce(results: Path, out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    lines = ["# Figures and tables", "", f"Regenerated from `{results}` by `reproduce.sh`.", ""]
    for name, (tables, source, fn) in ANALYSES.items():
        missing = [t for t in tables if not (results / t).is_file()]
        if missing:
            status[name] = "PENDING"
            lines.append(f"- **{name}**: [PENDING: run {source}] (missing {', '.join(missing)})")
            continue
        try:
            written = fn(results, out)
            status[name] = "ok"
            files = ", ".join(f"`{p.name}`" for p in written) or "(no series met the minimum size)"
            lines.append(f"- **{name}**: {files}")
        except InsufficientData as exc:
            status[name] = "insufficient data"
            lines.append(f"- **{name}**: insufficient data ({exc})")
        except Exception:  # noqa: BLE001  (record, keep regenerating the rest)
            status[name] = "error"
            lines.append(f"- **{name}**: ERROR\n\n```\n{traceback.format_exc()[-1500:]}\n```")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    for name, state in reproduce(a.results, a.out).items():
        print(f"{name}: {state}")


if __name__ == "__main__":
    main()
