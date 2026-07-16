"""CLI: score a metrics.json against a hardware profile.

    python -m edge_ai_compression.hardware.score \
        --metrics results/benchmark.json \
        --profile raspberry_pi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_ai_compression.hardware.profiles import available_profiles, get_profile
from edge_ai_compression.hardware.scoring import score_candidate


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metrics", type=Path, required=True, help="Path to a metrics JSON file.")
    p.add_argument(
        "--profile",
        default="raspberry_pi",
        help=f"Hardware profile. One of: {', '.join(available_profiles())}.",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional path to write the result JSON.")
    args = p.parse_args()

    if not args.metrics.is_file():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    # Accept a nested {"latency_ms": {...}} block from benchmark_model reports.
    if isinstance(metrics.get("latency_ms"), dict):
        metrics = {**metrics, "latency_ms": metrics["latency_ms"].get("mean")}

    profile = get_profile(args.profile)
    result = score_candidate(metrics, profile)

    print(f"Profile     : {profile.name}  ({profile.notes})")
    print(
        f"Budget      : latency<={profile.max_latency_ms:g}ms, "
        f"size<={profile.max_size_mb:g}MB, ram<={profile.max_ram_mb:g}MB"
    )
    print(f"Feasible    : {result.feasible}")
    print(f"Score       : {result.score:.4f}")
    print(f"Utilization : {result.utilization}")
    if result.violations:
        print("Violations  :")
        for v in result.violations:
            print(f"  - {v}")
    print(f"Explanation : {result.explanation}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
