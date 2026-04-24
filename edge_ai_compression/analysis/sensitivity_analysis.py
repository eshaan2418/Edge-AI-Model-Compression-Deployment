from __future__ import annotations

from typing import Any, Callable


def sweep_1d(
    values: list[float],
    evaluate: Callable[[float], dict[str, Any]],
    param_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for v in values:
        out = evaluate(v)
        rows.append({param_name: v, **out})
    return rows
