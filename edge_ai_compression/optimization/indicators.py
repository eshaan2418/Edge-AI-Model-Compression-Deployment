"""Quality indicators for a multi-objective (Pareto) front.

    python -m edge_ai_compression.optimization.indicators \
        --results results/experiments.csv --out reports/indicators.json

Finding the Pareto front tells you *which* trade-offs are non-dominated; it does
not tell you how *good* the front is. These indicators quantify that so you can
compare search methods or runs on a single number:

* **Hypervolume** — the volume of objective space dominated by the front relative
  to a reference (nadir) point. The single most-used quality indicator: larger is
  better, and it rewards both convergence and spread simultaneously.
* **Additive epsilon** — the smallest shift by which the front must be translated
  to dominate a reference front. Smaller is better (convergence).
* **Spacing** (Schott) — how evenly the front points are distributed. Smaller is
  more uniform.

Objectives are normalized to a common minimize-form using the project's keys
(maximize ``accuracy``; minimize ``latency_ms_mean``, ``size_mb``,
``peak_ram_mib``); maximize objectives are negated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MAXIMIZE = ("accuracy",)
DEFAULT_MINIMIZE = ("latency_ms_mean", "size_mb", "peak_ram_mib")


def _to_min_vector(point: dict[str, Any], maximize, minimize) -> list[float]:
    """Project a point onto a minimize-form objective vector (smaller = better)."""
    vec = [-float(point[k]) for k in maximize]
    vec += [float(point[k]) for k in minimize]
    return vec


def _nondominated(vectors: list[list[float]]) -> list[list[float]]:
    """Keep only minimize-form vectors not dominated by another (weakly)."""
    keep: list[list[float]] = []
    for i, v in enumerate(vectors):
        dominated = False
        for j, w in enumerate(vectors):
            if i == j:
                continue
            if all(w[k] <= v[k] for k in range(len(v))) and any(w[k] < v[k] for k in range(len(v))):
                dominated = True
                break
        if not dominated:
            keep.append(v)
    return keep


def hypervolume(front: list[list[float]], reference: list[float]) -> float:
    """Hypervolume of the region a minimize-form ``front`` dominates up to ``reference``.

    General-dimension recursive slicing: correct for any number of objectives.
    Only points componentwise ``<= reference`` contribute. Verified against
    hand-computed 2-D values in the tests.
    """
    d = len(reference)
    pts = [p for p in front if all(p[i] <= reference[i] for i in range(d))]
    if not pts:
        return 0.0
    if d == 1:
        return float(reference[0] - min(p[0] for p in pts))

    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in pts_sorted]
    volume = 0.0
    for k, x_lo in enumerate(xs):
        x_hi = xs[k + 1] if k + 1 < len(xs) else reference[0]
        width = x_hi - x_lo
        if width <= 0:
            continue
        # Cross-section in the remaining dims from all points starting at or
        # before this slab (their boxes extend upward from their own coord).
        projection = [p[1:] for p in pts_sorted[: k + 1]]
        volume += width * hypervolume(projection, reference[1:])
    return float(volume)


def epsilon_indicator(front: list[list[float]], reference_front: list[list[float]]) -> float:
    """Additive epsilon: min shift so ``front`` ε-dominates ``reference_front``.

    Both fronts are in minimize-form. Smaller is better (0 = front already
    dominates the reference front).
    """
    if not front or not reference_front:
        return float("inf")
    worst = float("-inf")
    for b in reference_front:
        best = float("inf")
        for a in front:
            eps = max(a[i] - b[i] for i in range(len(b)))
            best = min(best, eps)
        worst = max(worst, best)
    return float(worst)


def spacing(front: list[list[float]]) -> float:
    """Schott's spacing metric over minimize-form points (L1 nearest-neighbor)."""
    n = len(front)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        nn = min(
            sum(abs(front[i][k] - front[j][k]) for k in range(len(front[i])))
            for j in range(n)
            if j != i
        )
        dists.append(nn)
    mean = sum(dists) / n
    var = sum((d - mean) ** 2 for d in dists) / n
    return float(var**0.5)


def _auto_reference(vectors: list[list[float]]) -> list[float]:
    """Nadir point (worst per objective) plus a margin so volume is positive."""
    d = len(vectors[0])
    ref = []
    for i in range(d):
        col = [v[i] for v in vectors]
        hi, lo = max(col), min(col)
        margin = max(0.1 * (hi - lo), abs(hi) * 0.01, 1e-9)
        ref.append(hi + margin)
    return ref


def indicators(
    points: list[dict[str, Any]],
    *,
    maximize: tuple[str, ...] = DEFAULT_MAXIMIZE,
    minimize: tuple[str, ...] = DEFAULT_MINIMIZE,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute front-quality indicators over ``points``.

    Points missing any objective key are skipped. Returns hypervolume, spacing,
    the front size, and the reference point used (minimize-form).
    """
    keys = tuple(maximize) + tuple(minimize)
    usable = [p for p in points if all(k in p and p[k] is not None for k in keys)]
    if not usable:
        return {
            "num_points": 0,
            "front_size": 0,
            "hypervolume": 0.0,
            "spacing": 0.0,
            "reference": None,
        }

    vectors = [_to_min_vector(p, maximize, minimize) for p in usable]
    front = _nondominated(vectors)

    if reference is not None:
        ref = _to_min_vector(reference, maximize, minimize)
    else:
        ref = _auto_reference(vectors)

    return {
        "num_points": len(usable),
        "front_size": len(front),
        "hypervolume": hypervolume(front, ref),
        "spacing": spacing(front),
        "reference": ref,
        "objectives": {"maximize": list(maximize), "minimize": list(minimize)},
    }


def _load_points(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.optimization.indicators",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", type=Path, required=True, help="Results CSV or JSON.")
    p.add_argument("--maximize", default=",".join(DEFAULT_MAXIMIZE))
    p.add_argument("--minimize", default=",".join(DEFAULT_MINIMIZE))
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = p.parse_args(argv)

    if not args.results.is_file():
        print(f"Results file not found: {args.results}")
        return 2

    points = _load_points(args.results)
    maximize = tuple(x for x in args.maximize.split(",") if x)
    minimize = tuple(x for x in args.minimize.split(",") if x)
    report = indicators(points, maximize=maximize, minimize=minimize)

    if report["num_points"] == 0:
        print("No usable points (missing objective keys). Nothing to score.")
        return 1

    print(f"Points scored : {report['num_points']}")
    print(f"Front size    : {report['front_size']}")
    print(f"Hypervolume   : {report['hypervolume']:.6g}  (larger is better)")
    print(f"Spacing       : {report['spacing']:.6g}  (smaller is more uniform)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
