from __future__ import annotations

from collections.abc import Callable
from typing import Any


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
