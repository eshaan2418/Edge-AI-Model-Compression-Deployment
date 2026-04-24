from __future__ import annotations

"""Named compression pipelines for order-vs-Pareto studies (Phase 3)."""

CANONICAL_ORDER_TAGS: tuple[str, ...] = (
    "baseline",
    "prune",
    "quantize",
    "distill",
    "prune>quantize",
    "quantize>prune",
    "distill>quantize",
    "distill>prune",
    "prune>distill",
    "prune>distill>quantize",
    "distill>prune>quantize",
)


def parse_order(tag: str) -> list[str]:
    t = tag.strip().lower()
    if t == "baseline":
        return []
    if ">" in t:
        return [p.strip() for p in t.split(">") if p.strip()]
    return [t]


def format_order(parts: list[str]) -> str:
    return ">".join(parts) if parts else "baseline"
