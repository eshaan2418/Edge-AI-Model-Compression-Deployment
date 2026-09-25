"""Scaling of compression-induced accuracy loss with model size and training length.

Fits drop(x) = a * (x / x_min)^(-b) + c, where x is parameter count or training
epochs and drop is the final model's accuracy drop under a panel method. The
exponent b gets a 95% CI from bootstrapping runs (seeds) within each x value.
The fit is compared with a constant model by AICc; ``beats_constant`` is False
when the data does not support a trend, and is reported rather than hidden.

    python -m edge_ai_compression.analysis.scaling --results results --method w4a8
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from edge_ai_compression.analysis.common import read_table
from edge_ai_compression.experiments.compress_runs import method_tag


def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, -b) + c


def _aicc(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-300)
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float) / np.min(x)
    y = np.asarray(y, dtype=float)
    out = {"a": np.nan, "b": np.nan, "c": np.nan, "aicc_power": np.nan, "aicc_const": np.nan}
    out["aicc_const"] = _aicc(float(((y - y.mean()) ** 2).sum()), len(y), 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(
                power_law,
                x,
                y,
                p0=(y.max() - y.min() or 1e-3, 0.5, y.min()),
                bounds=([-np.inf, 0.0, -np.inf], [np.inf, 5.0, np.inf]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return {**out, "beats_constant": False}
    rss = float(((power_law(x, *p) - y) ** 2).sum())
    out.update(a=float(p[0]), b=float(p[1]), c=float(p[2]), aicc_power=_aicc(rss, len(y), 3))
    return {**out, "beats_constant": bool(out["aicc_power"] < out["aicc_const"])}


def scaling_points(
    results_dir: Path, method: str, x: str, filters: dict[str, object] | None = None
) -> pd.DataFrame:
    """Per-run (x, accuracy_drop, variant, seed) for final checkpoints under ``method``."""
    runs = read_table(results_dir / "training_runs.csv")
    exps = read_table(results_dir / "experiments.csv")
    prune_tag, quant_tag = method_tag(method)
    exps = exps[(exps["pruning_type"] == prune_tag) & (exps["quantization_type"] == quant_tag)]
    df = exps.merge(runs, left_on="source_run_id", right_on="run_id")
    df = df[df["source_step"] == df["steps"]]
    for key, value in (filters or {}).items():
        df = df[df[key] == value]
    out = df[["run_id", "variant", "seed", "model", "epochs", "num_params", "accuracy_drop"]].copy()
    out["x"] = df[x].astype(float)
    return out


def fit_with_bootstrap(points: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Power-law fit on per-x means, CI on b from resampling runs within each x."""
    means = points.groupby("x")["accuracy_drop"].mean()
    fit = fit_power_law(means.index.to_numpy(), means.to_numpy())
    rng = np.random.default_rng(seed)
    groups = {x: g["accuracy_drop"].to_numpy() for x, g in points.groupby("x")}
    xs = np.array(sorted(groups))
    bs = []
    for _ in range(n_boot):
        ys = np.array([rng.choice(groups[x], size=len(groups[x])).mean() for x in xs])
        b = fit_power_law(xs, ys)["b"]
        if np.isfinite(b):
            bs.append(b)
    lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
    return {**fit, "b_lo": float(lo), "b_hi": float(hi), "n_points": len(xs), "n_runs": len(points)}


def scaling_table(
    results_dir: Path,
    method: str,
    x: str,
    filters: dict[str, object] | None = None,
    n_boot: int = 500,
) -> pd.DataFrame:
    pts = scaling_points(results_dir, method, x, filters)
    rows = []
    for variant, g in pts.groupby("variant"):
        if g["x"].nunique() < 4:
            continue  # a 3-parameter fit needs at least 4 distinct x values
        rows.append({"method": method, "x": x, "variant": variant, **fit_with_bootstrap(g, n_boot)})
    return pd.DataFrame(rows)


def plot_scaling(points: pd.DataFrame, table: pd.DataFrame, path: Path, xlabel: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for variant, g in points.groupby("variant"):
        agg = g.groupby("x")["accuracy_drop"].agg(["mean", "std"]).reset_index()
        line = ax.errorbar(
            agg["x"],
            agg["mean"],
            yerr=agg["std"],
            marker="o",
            ms=3,
            ls="none",
            capsize=2,
            label=variant,
        )
        fit = table[table["variant"] == variant]
        if len(fit) and np.isfinite(fit.iloc[0]["b"]):
            f = fit.iloc[0]
            xs = np.geomspace(agg["x"].min(), agg["x"].max(), 100)
            ax.plot(
                xs,
                power_law(xs / agg["x"].min(), f["a"], f["b"], f["c"]),
                color=line[0].get_color(),
                lw=1,
            )
    ax.set(xscale="log", xlabel=xlabel, ylabel="Accuracy drop after compression")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--method", default="w4a8")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    studies = {
        "params": ("num_params", {"epochs": 30}, "Parameters"),
        "epochs": ("epochs", {"model": "resnet18_w0.5_cifar", "variant": "standard"}, "Epochs"),
    }
    for name, (x, filters, label) in studies.items():
        table = scaling_table(a.results, a.method, x, filters)
        table.to_csv(a.out / f"scaling_{name}_{a.method}.csv", index=False)
        pts = scaling_points(a.results, a.method, x, filters)
        print(plot_scaling(pts, table, a.out / f"scaling_{name}_{a.method}.png", label))


if __name__ == "__main__":
    main()
