"""Roofline / arithmetic-intensity analysis for a model on a hardware profile.

    python -m edge_ai_compression.theory.roofline \
        --model resnet18_cifar --profile raspberry_pi --measured-latency-ms 180

The roofline model (Williams et al., 2009) bounds achievable compute throughput
by the smaller of two ceilings: the hardware's peak FLOP/s, and its memory
bandwidth times the workload's *arithmetic intensity* (FLOPs per byte moved). A
workload is compute-bound above the "ridge" intensity and memory-bound below it.

From that we derive a **theoretical latency lower bound** — the fastest the model
could run if it hit either ceiling perfectly. Real latency is always slower; the
ratio (lower_bound / measured) is a rough "efficiency" telling you how much of the
gap is the hardware ceiling versus software overhead.

All hardware numbers come from the profile's *nominal* peak specs (theoretical
ceilings, not measurements) — see ``hardware.profiles``. FLOPs are counted as
MACs x 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.hardware.profiles import HardwareProfile, get_profile
from edge_ai_compression.theory.model_complexity import (
    estimate_activation_bytes,
    estimate_flops_macs,
    weight_bytes,
)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic. Higher = more compute-bound."""
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def roofline(
    intensity: float,
    *,
    peak_gflops: float,
    mem_bandwidth_gbs: float,
) -> dict[str, Any]:
    """Attainable throughput and the compute/memory-bound classification.

    ``attainable = min(peak_gflops, mem_bandwidth_gbs * intensity)`` (both in
    GFLOP/s, since GB/s x FLOP/byte = GFLOP/s). The ridge point is the intensity
    at which the two ceilings meet.
    """
    if peak_gflops <= 0 or mem_bandwidth_gbs <= 0:
        raise ValueError("peak_gflops and mem_bandwidth_gbs must be positive")
    memory_ceiling = mem_bandwidth_gbs * intensity
    attainable = min(peak_gflops, memory_ceiling)
    ridge = peak_gflops / mem_bandwidth_gbs
    return {
        "attainable_gflops": attainable,
        "ridge_intensity": ridge,
        "bound": "compute" if intensity >= ridge else "memory",
    }


def theoretical_min_latency_ms(
    flops: float,
    bytes_moved: float,
    *,
    peak_gflops: float,
    mem_bandwidth_gbs: float,
) -> dict[str, float]:
    """Latency lower bounds (ms) from the compute and memory ceilings.

    The roofline lower bound is the *larger* of the two — you cannot finish faster
    than either the compute or the memory transfer allows.
    """
    compute_ms = flops / (peak_gflops * 1e9) * 1000.0
    memory_ms = bytes_moved / (mem_bandwidth_gbs * 1e9) * 1000.0
    return {
        "compute_bound_ms": compute_ms,
        "memory_bound_ms": memory_ms,
        "lower_bound_ms": max(compute_ms, memory_ms),
    }


def analyze(
    model,
    input_shape: tuple[int, int, int, int],
    profile: HardwareProfile,
    *,
    measured_latency_ms: float | None = None,
    weight_dtype_bytes: int = 4,
    activation_dtype_bytes: int = 4,
) -> dict[str, Any]:
    """Full roofline report for ``model`` on ``profile``.

    Raises ValueError if the profile lacks nominal compute/bandwidth ceilings.
    """
    if profile.peak_gflops is None or profile.mem_bandwidth_gbs is None:
        raise ValueError(
            f"Profile '{profile.name}' has no nominal peak_gflops / mem_bandwidth_gbs; "
            "roofline analysis needs both."
        )

    macs = estimate_flops_macs(model, input_shape)
    flops = macs * 2.0
    w_bytes = weight_bytes(model, dtype_bytes=weight_dtype_bytes)
    a_bytes = estimate_activation_bytes(model, input_shape, dtype_bytes=activation_dtype_bytes)
    bytes_moved = w_bytes + a_bytes

    intensity = arithmetic_intensity(flops, bytes_moved)
    rl = roofline(
        intensity,
        peak_gflops=profile.peak_gflops,
        mem_bandwidth_gbs=profile.mem_bandwidth_gbs,
    )
    bounds = theoretical_min_latency_ms(
        flops,
        bytes_moved,
        peak_gflops=profile.peak_gflops,
        mem_bandwidth_gbs=profile.mem_bandwidth_gbs,
    )

    report: dict[str, Any] = {
        "profile": profile.name,
        "peak_gflops": profile.peak_gflops,
        "mem_bandwidth_gbs": profile.mem_bandwidth_gbs,
        "flops": flops,
        "macs": macs,
        "weight_bytes": w_bytes,
        "activation_bytes": a_bytes,
        "bytes_moved": bytes_moved,
        "arithmetic_intensity": intensity,
        "roofline": rl,
        "latency_bounds_ms": bounds,
        "measured_latency_ms": measured_latency_ms,
        "note": (
            "Hardware ceilings are nominal peak specs, not measurements; "
            "latency bounds are theoretical floors (real latency is slower)."
        ),
    }
    if measured_latency_ms is not None and measured_latency_ms > 0:
        report["roofline_efficiency"] = bounds["lower_bound_ms"] / measured_latency_ms
    return report


def _print_report(report: dict[str, Any]) -> None:
    rl = report["roofline"]
    b = report["latency_bounds_ms"]
    print(f"Profile              : {report['profile']}")
    print(f"  peak               : {report['peak_gflops']:g} GFLOP/s (nominal)")
    print(f"  bandwidth          : {report['mem_bandwidth_gbs']:g} GB/s (nominal)")
    print(f"FLOPs                : {report['flops']:,.0f}")
    print(
        f"Bytes moved          : {report['bytes_moved']:,.0f} "
        f"(weights {report['weight_bytes']:,.0f} + activations {report['activation_bytes']:,.0f})"
    )
    print(f"Arithmetic intensity : {report['arithmetic_intensity']:.3f} FLOP/byte")
    print(f"Ridge intensity      : {rl['ridge_intensity']:.3f} FLOP/byte")
    print(
        f"Regime               : {rl['bound']}-bound "
        f"(attainable {rl['attainable_gflops']:.2f} GFLOP/s)"
    )
    print(
        f"Latency lower bound  : {b['lower_bound_ms']:.4f} ms "
        f"(compute {b['compute_bound_ms']:.4f} / memory {b['memory_bound_ms']:.4f})"
    )
    if report.get("measured_latency_ms"):
        print(f"Measured latency     : {report['measured_latency_ms']:.4f} ms")
        print(f"Roofline efficiency  : {report['roofline_efficiency'] * 100:.1f}% of the floor")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.theory.roofline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="resnet18_cifar", help="Registry model name.")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--input-shape", default="1,3,32,32", help="Comma-separated, e.g. 1,3,32,32.")
    p.add_argument("--profile", default="raspberry_pi", help="Hardware profile name.")
    p.add_argument("--measured-latency-ms", type=float, default=None)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = p.parse_args(argv)

    if args.model not in ModelRegistry.available():
        print(f"Unknown model '{args.model}'. Available: {', '.join(ModelRegistry.available())}")
        return 1
    try:
        profile = get_profile(args.profile)
    except KeyError as exc:
        print(str(exc))
        return 1
    try:
        shape = tuple(int(x) for x in args.input_shape.split(","))
    except ValueError:
        print(f"Invalid --input-shape: {args.input_shape}")
        return 1

    model = ModelRegistry.create(args.model, num_classes=args.num_classes)
    try:
        report = analyze(model, shape, profile, measured_latency_ms=args.measured_latency_ms)
    except ValueError as exc:
        print(str(exc))
        return 1

    _print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
