from __future__ import annotations

from typing import Any


def dominates(
    a: dict[str, Any],
    b: dict[str, Any],
    metrics_maximize: tuple[str, ...],
    metrics_minimize: tuple[str, ...],
) -> bool:
    """True if a Pareto-dominates b.

    Dominance means a is not worse than b in any objective and strictly better
    in at least one.
    """
    for k in metrics_maximize:
        if float(a[k]) < float(b[k]):
            return False
    for k in metrics_minimize:
        if float(a[k]) > float(b[k]):
            return False
    strictly_better = any(float(a[k]) > float(b[k]) for k in metrics_maximize) or any(
        float(a[k]) < float(b[k]) for k in metrics_minimize
    )
    return strictly_better
