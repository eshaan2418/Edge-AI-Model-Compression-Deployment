"""Small-LM track: does compression degrade post-trained behavior before capability?

Per (model, seed, stage), every ladder rung is normalized to that stage's float
model: relative capability = ppl_fp / ppl, relative behavior = rate / rate_fp
(both 1 at float, lower = worse). The per-seed gap (capability - behavior) is
aggregated over seeds with a bootstrap CI; a gap whose CI lies above 0 means
behavior degraded more than capability at that rung.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edge_ai_compression.analysis.common import read_table
from edge_ai_compression.benchmarking.stats import bootstrap_ci
from edge_ai_compression.lm.compress import LM_LADDER


def _ci(values: pd.Series) -> tuple[float, float, float]:
    v = values.dropna().to_numpy(float)
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size < 2:
        return float(v.mean()), float("nan"), float("nan")
    lo, hi = bootstrap_ci(v, np.mean, n_boot=2000)
    return float(v.mean()), lo, hi


def relative_rows(evals: pd.DataFrame) -> pd.DataFrame:
    df = evals[evals["method"] != "speculative"]
    out = []
    for (model, seed, stage), g in df.groupby(["model", "seed", "stage"]):
        fp = g[g["method"] == "fp"]
        if fp.empty:
            continue
        ppl0, rate0 = float(fp["perplexity"].iloc[0]), fp["constraint_rate"].iloc[0]
        for r in g.itertuples():
            rel_cap = ppl0 / float(r.perplexity)
            rel_beh = (
                float(r.constraint_rate) / float(rate0)
                if pd.notna(rate0) and float(rate0) > 0 and pd.notna(r.constraint_rate)
                else float("nan")
            )
            out.append(
                {
                    "model": model,
                    "seed": seed,
                    "stage": stage,
                    "method": r.method,
                    "rel_capability": rel_cap,
                    "rel_behavior": rel_beh,
                    "gap": rel_cap - rel_beh,
                }
            )
    return pd.DataFrame(out)


def degradation_table(results_dir: Path) -> pd.DataFrame:
    rel = relative_rows(read_table(results_dir / "lm_evals.csv"))
    rows = []
    order = {m: i for i, m in enumerate(LM_LADDER)}
    for (model, stage, method), g in rel.groupby(["model", "stage", "method"]):
        cap, beh, gap = _ci(g["rel_capability"]), _ci(g["rel_behavior"]), _ci(g["gap"])
        rows.append(
            {
                "model": model,
                "stage": stage,
                "method": method,
                "n_seeds": len(g),
                "rel_capability": cap[0],
                "rel_capability_lo": cap[1],
                "rel_capability_hi": cap[2],
                "rel_behavior": beh[0],
                "rel_behavior_lo": beh[1],
                "rel_behavior_hi": beh[2],
                "gap": gap[0],
                "gap_lo": gap[1],
                "gap_hi": gap[2],
                "behavior_degrades_first": bool(np.isfinite(gap[1]) and gap[1] > 0),
                "_order": order.get(method, len(order)),
            }
        )
    table = pd.DataFrame(rows)
    return (
        table.sort_values(["model", "stage", "_order"]).drop(columns="_order")
        if len(table)
        else table
    )


def plot_degradation(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(k, g) for k, g in table.groupby(["model", "stage"]) if k[1] != "pretrain"]
    fig, axes = plt.subplots(
        1, max(len(groups), 1), figsize=(4 * max(len(groups), 1), 3.5), squeeze=False
    )
    for ax, ((model, stage), g) in zip(axes[0], groups, strict=False):
        x = np.arange(len(g))
        for col, label in (
            ("rel_capability", "capability (ppl_fp / ppl)"),
            ("rel_behavior", "behavior (rate / rate_fp)"),
        ):
            ax.plot(x, g[col], marker="o", ms=3, label=label)
            ax.fill_between(x, g[f"{col}_lo"], g[f"{col}_hi"], alpha=0.15)
        ax.set_xticks(x, g["method"], rotation=45, fontsize=7)
        ax.set(title=f"{model} / {stage}", ylabel="relative to float")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
