"""CLI: score one set of metrics against ALL hardware profiles.

    python -m edge_ai_compression.hardware.compare \
        --metrics results/demo/report.json \
        --out results/hardware_comparison.md

Reads a demo report, benchmark report, or plain metrics JSON, scores it against
every profile, and prints a Markdown feasibility table (also written to --out).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edge_ai_compression.hardware.profiles import available_profiles, get_profile
from edge_ai_compression.hardware.scoring import score_candidate
from edge_ai_compression.utils.metrics_io import extract_metrics, load_json


def compare_all_profiles(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Score ``metrics`` (canonical or raw) against every profile.

    Returns one row per profile, sorted feasible-first then by score desc.
    """
    canonical = extract_metrics(metrics) if _looks_like_report(metrics) else metrics
    score_input = {
        "accuracy": canonical.get("accuracy"),
        "latency_ms": canonical.get("latency_ms"),
        "size_mb": canonical.get("size_mb"),
        "ram_mb": canonical.get("ram_mb"),
    }

    rows: list[dict[str, Any]] = []
    for name in available_profiles():
        profile = get_profile(name)
        result = score_candidate(score_input, profile)
        rows.append(
            {
                "profile": name,
                "feasible": result.feasible,
                "score": result.score,
                "num_violations": len(result.violations),
                "violations": result.violations,
                "preferred_export": profile.preferred_export_format,
            }
        )
    rows.sort(key=lambda r: (not r["feasible"], -r["score"]))
    return rows


def _looks_like_report(data: dict[str, Any]) -> bool:
    """True if this is a report shape we should run through extract_metrics."""
    return "compressed" in data or isinstance(data.get("latency_ms"), dict)


def to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Hardware profile comparison",
        "",
        "| Profile | Feasible | Score | Violations | Preferred export |",
        "| ------- | -------- | ----- | ---------- | ---------------- |",
    ]
    for r in rows:
        feasible = "✅" if r["feasible"] else "❌"
        lines.append(
            f"| {r['profile']} | {feasible} | {r['score']:.4f} | "
            f"{r['num_violations']} | {r['preferred_export']} |"
        )
    return "\n".join(lines) + "\n"


def _print_console(rows: list[dict[str, Any]]) -> None:
    print(f"{'Profile':<20} {'Feasible':<9} {'Score':>10} {'Violations':>11}  Preferred")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['profile']:<20} {str(r['feasible']):<9} {r['score']:>10.4f} "
            f"{r['num_violations']:>11}  {r['preferred_export']}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.hardware.compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics", type=Path, required=True, help="Metrics/report JSON.")
    p.add_argument("--out", type=Path, default=None, help="Optional Markdown output path.")
    args = p.parse_args(argv)

    if not args.metrics.is_file():
        print(f"Metrics file not found: {args.metrics}")
        return 2

    metrics = load_json(args.metrics)
    rows = compare_all_profiles(metrics)
    _print_console(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_markdown(rows), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
