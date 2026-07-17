"""Statistically compare the inference latency of two models.

    python -m edge_ai_compression.benchmarking.ab_compare \
        --model-a small_cnn_student --model-b resnet18_cifar \
        --repeats 100 --out reports/ab_latency.json

Builds both models from the registry, measures per-run latency on synthetic input,
and reports bootstrap confidence intervals, a Mann-Whitney U significance test, a
Cliff's-delta effect size, and a bootstrap CI on the speedup factor — then prints a
plain-English verdict about whether the difference is real or just noise.

Latency is wall-clock on this host, not the target device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from edge_ai_compression.benchmarking.latency_profiler import profile_model_detailed
from edge_ai_compression.benchmarking.statistics import compare_latency
from edge_ai_compression.core.registry import ModelRegistry


def _measure(
    model_name: str,
    *,
    num_classes: int,
    shape: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> list[float]:
    model = ModelRegistry.create(model_name, num_classes=num_classes)
    sample = torch.randn(*shape)
    detailed = profile_model_detailed(model, sample, warmup=warmup, repeats=repeats)
    return detailed["trace_ms"]


def _row(label: str, stats: dict, cv: float) -> str:
    ci = f"[{stats['low']:.3f}, {stats['high']:.3f}]"
    return f"{label:<22}{stats['point']:>12.3f}{ci:>24}{cv:>8.3f}"


def _print_verdict(report: dict) -> None:
    a, b = report["label_a"], report["label_b"]
    su = report["speedup_b_over_a"]
    print(f"{'Model':<22}{'median ms':>12}{'95% CI':>24}{'CV':>8}")
    print("-" * 66)
    print(_row(a, report[a]["median_ms"], report[a]["cv"]))
    print(_row(b, report[b]["median_ms"], report[b]["cv"]))
    print("-" * 66)
    print(f"speedup ({b} over {a}): {su['point']:.3f}x  [{su['low']:.3f}, {su['high']:.3f}]")
    print(f"\n{report['verdict']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.benchmarking.ab_compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    names = ", ".join(ModelRegistry.available())
    p.add_argument("--model-a", required=True, help=f"One of: {names}.")
    p.add_argument("--model-b", required=True, help="Second model to compare against.")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--input-shape", default="1,3,32,32", help="Comma-separated, e.g. 1,3,32,32.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level for the test.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = p.parse_args(argv)

    available = ModelRegistry.available()
    for name in (args.model_a, args.model_b):
        if name not in available:
            print(f"Unknown model '{name}'. Available: {', '.join(available)}")
            return 1

    try:
        shape = tuple(int(x) for x in args.input_shape.split(","))
    except ValueError:
        print(f"Invalid --input-shape: {args.input_shape}")
        return 1

    kw = {
        "num_classes": args.num_classes,
        "shape": shape,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    trace_a = _measure(args.model_a, **kw)
    trace_b = _measure(args.model_b, **kw)

    report = compare_latency(
        trace_a,
        trace_b,
        label_a=args.model_a,
        label_b=args.model_b,
        confidence=args.confidence,
        alpha=args.alpha,
        seed=args.seed,
    )
    _print_verdict(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
