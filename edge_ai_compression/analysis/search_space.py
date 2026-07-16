"""Summarize a compression/search space before running an expensive sweep.

    python -m edge_ai_compression.analysis.search_space \
        --config configs/sweeps/full_compression_study.yml \
        --out reports/search_space.md

Counts candidate combinations, lists each dimension and its values, and warns
when the space is large — so you can gauge the cost of a sweep before launching
it. Handles three shapes:

* explicit ``variants`` lists (count = number of variants);
* grid dicts where each dimension maps to a list (cartesian product);
* distribution specs (``type: choice`` is enumerable; ``type: uniform`` etc. are
  continuous and cannot be enumerated).
"""

from __future__ import annotations

import argparse
from math import prod
from pathlib import Path
from typing import Any

import yaml

# Keys that describe the sweep itself, not a search dimension.
_META_KEYS = {"sweep_id", "base_config", "variants", "name", "description"}
_LARGE_SPACE_WARNING = 100


def _dimension_from_spec(name: str, spec: Any) -> dict[str, Any]:
    """Classify one dimension spec into enumerated/continuous/fixed."""
    if isinstance(spec, list):
        return {"name": name, "kind": "enumerated", "values": spec, "count": len(spec)}
    if isinstance(spec, dict):
        kind = str(spec.get("type", "")).lower()
        if "values" in spec or "choices" in spec or kind == "choice":
            values = spec.get("values") or spec.get("choices") or []
            return {"name": name, "kind": "enumerated", "values": values, "count": len(values)}
        # Continuous distribution (uniform / loguniform / normal / ...).
        rng = {k: spec[k] for k in ("low", "high", "mean", "std") if k in spec}
        return {"name": name, "kind": "continuous", "range": rng, "count": None}
    # A plain scalar is a fixed value, not a real search dimension.
    return {"name": name, "kind": "fixed", "value": spec, "count": 1}


def summarize_search_space(config: dict[str, Any]) -> dict[str, Any]:
    """Return a structured summary of the search space in ``config``."""
    if not isinstance(config, dict):
        raise ValueError("Search-space config must be a mapping.")

    variants = config.get("variants")
    if isinstance(variants, list):
        # Explicit variant list: dimensions are the union of keys across variants.
        dims: dict[str, list[Any]] = {}
        for v in variants:
            if isinstance(v, dict):
                for k, val in v.items():
                    dims.setdefault(k, [])
                    if val not in dims[k]:
                        dims[k].append(val)
        dimensions = [
            {"name": k, "kind": "enumerated", "values": vals, "count": len(vals)}
            for k, vals in dims.items()
        ]
        return {
            "mode": "explicit_variants",
            "num_candidates": len(variants),
            "dimensions": dimensions,
            "base_config": config.get("base_config"),
            "has_continuous": False,
        }

    # Otherwise treat non-meta keys as dimension specs.
    dimensions = [
        _dimension_from_spec(name, spec) for name, spec in config.items() if name not in _META_KEYS
    ]
    enumerated = [d for d in dimensions if d["kind"] == "enumerated"]
    has_continuous = any(d["kind"] == "continuous" for d in dimensions)
    num_candidates = (
        None if has_continuous or not enumerated else prod(d["count"] for d in enumerated)
    )
    return {
        "mode": "grid",
        "num_candidates": num_candidates,
        "dimensions": dimensions,
        "base_config": config.get("base_config"),
        "has_continuous": has_continuous,
    }


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Search space summary",
        "",
        f"- Mode: `{summary['mode']}`",
    ]
    if summary.get("base_config"):
        lines.append(f"- Base config: `{summary['base_config']}`")

    n = summary["num_candidates"]
    if n is None:
        lines.append("- Candidate count: **continuous / unbounded** (contains sampled dimensions)")
    else:
        lines.append(f"- Candidate count: **{n}**")
        if n >= _LARGE_SPACE_WARNING:
            lines.append(f"- ⚠️ Large space (≥ {_LARGE_SPACE_WARNING}); a full grid may be costly.")
    lines += ["", "## Dimensions", ""]

    if not summary["dimensions"]:
        lines.append("_No search dimensions found._")
        return "\n".join(lines) + "\n"

    lines += ["| Dimension | Kind | Values / range | Count |", "| --- | --- | --- | --- |"]
    for d in summary["dimensions"]:
        if d["kind"] == "enumerated":
            detail = ", ".join(str(v) for v in d["values"])
        elif d["kind"] == "continuous":
            detail = str(d.get("range", {}))
        else:
            detail = str(d.get("value"))
        count = "∞" if d["count"] is None else str(d["count"])
        lines.append(f"| {d['name']} | {d['kind']} | {detail} | {count} |")
    return "\n".join(lines) + "\n"


def _print_console(summary: dict[str, Any]) -> None:
    n = summary["num_candidates"]
    count_str = "continuous/unbounded" if n is None else str(n)
    print(f"Mode           : {summary['mode']}")
    print(f"Candidate count: {count_str}")
    if isinstance(n, int) and n >= _LARGE_SPACE_WARNING:
        print(f"WARNING        : large space (>= {_LARGE_SPACE_WARNING}); full grid may be costly.")
    for d in summary["dimensions"]:
        count = "inf" if d["count"] is None else d["count"]
        print(f"  - {d['name']} ({d['kind']}, {count})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.analysis.search_space",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Sweep / search-space YAML.")
    p.add_argument("--out", type=Path, default=None, help="Optional Markdown output path.")
    args = p.parse_args(argv)

    if not args.config.is_file():
        print(f"Config file not found: {args.config}")
        return 2

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    summary = summarize_search_space(config)
    _print_console(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_markdown(summary), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
