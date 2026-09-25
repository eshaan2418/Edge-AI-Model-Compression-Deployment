"""Early predictability of compressibility from training-time signals (Phase 6).

For a compression method M (a panel entry from ``compress_runs``) and a training
fraction f, each training run contributes one row:
  features = signals logged nearest to step f * total_steps (+ log #params and a
             variant one-hot)
  target   = accuracy drop of the run's *final* model under M
Predictors are evaluated with leave-one-configuration-out cross-validation
(configuration = model x variant x options x epochs; all seeds held out together,
DECISIONS D6.1). Metrics (Spearman rho, R^2, MAE) get 95% CIs from a cluster
bootstrap over configurations, since seeds of one configuration are correlated.

    python -m edge_ai_compression.analysis.early_prediction --results results --method w4a8
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edge_ai_compression.experiments.compress_runs import method_tag
from edge_ai_compression.pretraining.signals import SCALAR_FIELDS

FRACTIONS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "weights": (
        "kurtosis_mean",
        "kurtosis_max",
        "w_outlier_mean",
        "w_outlier_max",
        "weight_l2_mean",
    ),
    "activations": (
        "act_outlier_mean",
        "act_outlier_max",
        "act_channel_ratio_mean",
        "act_channel_ratio_max",
    ),
    "curvature": ("hessian_trace",),
    "sharpness": ("sharpness",),
    "loss": ("probe_loss",),
    "meta": ("log_params",),  # variant one-hot columns are added to "meta" dynamically
}

PREDICTORS: dict[str, Callable[[], RegressorMixin]] = {
    "mean": lambda: DummyRegressor(strategy="mean"),
    "params_only": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "rf": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    "gbm": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    ),
    "mlp": lambda: make_pipeline(
        StandardScaler(), MLPRegressor((32, 16), alpha=1e-2, max_iter=3000, random_state=0)
    ),
    "gp": lambda: make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(), normalize_y=True, random_state=0),
    ),
}


def load_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "training_runs.csv")
    signals = pd.read_csv(results_dir / "training_signals.csv")
    exps = pd.read_csv(results_dir / "experiments.csv")
    return runs, signals, exps


def build_dataset(
    runs: pd.DataFrame, signals: pd.DataFrame, exps: pd.DataFrame, method: str, fraction: float
) -> pd.DataFrame:
    """One row per training run that has both signals and a final-model target for ``method``."""
    prune_tag, quant_tag = method_tag(method)
    target = exps[(exps["pruning_type"] == prune_tag) & (exps["quantization_type"] == quant_tag)]
    target = target.merge(runs[["run_id", "steps"]], left_on="source_run_id", right_on="run_id")
    target = target[target["source_step"] == target["steps"]]  # final checkpoint only
    target = target.groupby("source_run_id").agg(
        accuracy_drop=("accuracy_drop", "mean"), num_params=("num_params", "first")
    )
    rows = []
    for run in runs.itertuples():
        if run.run_id not in target.index:
            continue
        sig = signals[signals["run_id"] == run.run_id]
        if sig.empty:
            continue
        nearest = sig.iloc[(sig["step"] - fraction * run.steps).abs().argsort().iloc[0]]
        rows.append(
            {
                "run_id": run.run_id,
                "config": "|".join(
                    str(v)
                    for v in (
                        run.model,
                        run.variant,
                        getattr(run, "variant_options", ""),
                        run.epochs,
                    )
                ),
                "variant": run.variant,
                "log_params": float(np.log10(target.loc[run.run_id, "num_params"])),
                "signal_step": int(nearest["step"]),
                **{k: float(nearest[k]) for k in SCALAR_FIELDS},
                "target": float(target.loc[run.run_id, "accuracy_drop"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for v in sorted(df["variant"].unique()):
        df[f"variant_{v}"] = (df["variant"] == v).astype(float)
    return df


def feature_columns(df: pd.DataFrame, groups: tuple[str, ...] | None = None) -> list[str]:
    groups = groups or tuple(FEATURE_GROUPS)
    cols: list[str] = []
    for g in groups:
        cols += [c for c in FEATURE_GROUPS[g] if c in df]
        if g == "meta":
            cols += [c for c in df if c.startswith("variant_")]
    # Drop columns that are constant or NaN everywhere (e.g. a disabled signal).
    return [c for c in cols if df[c].notna().all() and df[c].nunique() > 1]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, p).statistic if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {
        "spearman": float(rho),
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
    }


def evaluate(
    df: pd.DataFrame, predictor: str, columns: list[str], n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Leave-one-configuration-out predictions + cluster-bootstrap CIs over configurations."""
    groups = df["config"].to_numpy()
    y = df["target"].to_numpy()
    x = df[columns].to_numpy() if columns else np.zeros((len(df), 1))
    if predictor == "params_only":
        x = df[["log_params"]].to_numpy()
    pred = np.empty_like(y)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = PREDICTORS[predictor]()
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    out = {**_metrics(y, pred), "n_runs": len(df), "n_configs": len(set(groups))}
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    boots: dict[str, list[float]] = {"spearman": [], "r2": [], "mae": []}
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, size=len(uniq))])
        for k, v in _metrics(y[pick], pred[pick]).items():
            boots[k].append(v)
    for k, vals in boots.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.quantile(arr, [0.025, 0.975]) if arr.size else (np.nan, np.nan)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return out


def prediction_vs_fraction(
    results_dir: Path,
    method: str,
    predictors: tuple[str, ...] = tuple(PREDICTORS),
    fractions: tuple[float, ...] = FRACTIONS,
    n_boot: int = 1000,
) -> pd.DataFrame:
    runs, signals, exps = load_tables(results_dir)
    rows = []
    for f in fractions:
        df = build_dataset(runs, signals, exps, method, f)
        if df["config"].nunique() < 3:
            raise ValueError(
                f"need >= 3 configurations for grouped CV, have {df['config'].nunique()}"
            )
        cols = feature_columns(df)
        for name in predictors:
            rows.append(
                {
                    "method": method,
                    "fraction": f,
                    "predictor": name,
                    **evaluate(df, name, cols, n_boot),
                }
            )
    return pd.DataFrame(rows)


def ablation(df: pd.DataFrame, predictor: str = "gbm", n_boot: int = 1000) -> pd.DataFrame:
    """Each feature group alone, and all features minus each group."""
    rows = []
    for g in FEATURE_GROUPS:
        only = feature_columns(df, (g,))
        rest = feature_columns(df, tuple(x for x in FEATURE_GROUPS if x != g))
        if only:
            rows.append({"group": g, "setting": "only", **evaluate(df, predictor, only, n_boot)})
        if rest:
            rows.append({"group": g, "setting": "without", **evaluate(df, predictor, rest, n_boot)})
    return pd.DataFrame(rows)


def plot_vs_fraction(table: pd.DataFrame, path: Path, metric: str = "spearman") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in table.groupby("predictor"):
        g = g.sort_values("fraction")
        x = g["fraction"].clip(lower=1e-3)
        ax.plot(x, g[metric], marker="o", ms=3, label=name)
        ax.fill_between(x, g[f"{metric}_lo"], g[f"{metric}_hi"], alpha=0.15)
    ax.set(
        xscale="log",
        xlabel="Training fraction at which signals are read (0 plotted at 1e-3)",
        ylabel=f"{metric} (leave-one-configuration-out)",
    )
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
    table = prediction_vs_fraction(a.results, a.method)
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / f"early_prediction_{a.method}.csv", index=False)
    runs, signals, exps = load_tables(a.results)
    for f in (0.1, 1.0):
        ablation(build_dataset(runs, signals, exps, a.method, f)).to_csv(
            a.out / f"ablation_{a.method}_f{f}.csv", index=False
        )
    print(plot_vs_fraction(table, a.out / f"early_prediction_{a.method}.png"))


if __name__ == "__main__":
    main()
