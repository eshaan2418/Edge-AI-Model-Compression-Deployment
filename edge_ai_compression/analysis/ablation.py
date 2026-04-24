from __future__ import annotations

from typing import Any, Callable


def run_ablation(experiments: list[tuple[str, Callable[[], dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Run named ablation arms; each callable returns metrics dict."""
    return [{"arm": name, **fn()} for name, fn in experiments]
