"""Benchmark a model on synthetic input and emit a JSON report.

CPU-only and fully offline: input tensors are random, so this measures the
model's *compute* characteristics (latency distribution, size, parameters)
without any dataset. For accuracy on real data use the experiment runner.

    python -m edge_ai_compression.benchmarking.benchmark_model \
        --config edge_ai_compression/configs/experiments/smoke_cpu.yml \
        --out results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from edge_ai_compression.benchmarking.latency_profiler import profile_model_detailed
from edge_ai_compression.benchmarking.memory_profiler import estimate_peak_rss_mib
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.theory.model_complexity import estimate_flops_macs, model_summary
from edge_ai_compression.utils.metrics import compute_latency_stats


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    *,
    warmup: int = 5,
    repeats: int = 50,
    measure_memory: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Benchmark ``model`` on a synthetic input of ``input_shape``.

    Returns a JSON-serializable dict with latency percentiles (ms), throughput,
    parameter/size/FLOP estimates, and (optionally) peak RSS.
    """
    model = model.to(device).eval()
    sample = torch.randn(*input_shape, device=device)

    det = profile_model_detailed(model, sample, warmup=warmup, repeats=repeats)
    stats = compute_latency_stats(det["trace_ms"])

    summary = model_summary(model)
    report: dict[str, Any] = {
        "input_shape": list(input_shape),
        "device": device,
        "warmup_iters": warmup,
        "measured_iters": repeats,
        "latency_ms": {
            "mean": stats["mean"],
            "std": stats["std"],
            "p50": stats["p50"],
            "p95": stats["p95"],
            "p99": stats["p99"],
            "cold_start_ms": float(det["cold_start_ms"]),
        },
        "throughput_ips": float(det["throughput_ips"]),
        "num_parameters": summary["num_parameters"],
        "num_trainable_parameters": summary["num_trainable_parameters"],
        "size_mb": summary["size_mb"],
        "weight_sparsity": summary["weight_sparsity"],
        "flops_estimate": estimate_flops_macs(model, input_shape),
    }

    if measure_memory:

        def forward_once() -> None:
            with torch.no_grad():
                model(sample)

        report["peak_ram_mib"] = float(estimate_peak_rss_mib(forward_once))

    return report


def _build_from_config(config_path: Path) -> tuple[torch.nn.Module, tuple[int, int, int, int]]:
    from edge_ai_compression.core.runner import load_experiment_config

    cfg = load_experiment_config(config_path)
    model = ModelRegistry.create(cfg.model, num_classes=cfg.num_classes)
    shape = cfg.benchmark.get("input_shape", [1, 3, 32, 32])
    input_shape = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    return model, input_shape


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--config", type=Path, help="Experiment YAML (model + input_shape).")
    src.add_argument("--model", help="Model name from the registry (e.g. resnet18_cifar).")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--no-memory", action="store_true", help="Skip peak-RAM measurement.")
    p.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    args = p.parse_args()

    if args.config:
        model, input_shape = _build_from_config(args.config)
        source = str(args.config)
    else:
        model_name = args.model or "resnet18_cifar"
        model = ModelRegistry.create(model_name, num_classes=args.num_classes)
        input_shape = (1, 3, 32, 32)
        source = model_name

    report = benchmark_model(
        model,
        input_shape,
        warmup=args.warmup,
        repeats=args.repeats,
        measure_memory=not args.no_memory,
    )
    report["source"] = source

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lat = report["latency_ms"]
    print(f"Benchmarked: {source}")
    print(f"  params      : {report['num_parameters']:,}")
    print(f"  size        : {report['size_mb']:.3f} MB")
    print(f"  latency mean: {lat['mean']:.3f} ms  (p95 {lat['p95']:.3f}, p99 {lat['p99']:.3f})")
    print(f"  throughput  : {report['throughput_ips']:.1f} inf/s")
    print(f"  report      -> {args.out}")


if __name__ == "__main__":
    main()
