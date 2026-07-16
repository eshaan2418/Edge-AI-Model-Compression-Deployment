"""Offline, self-contained compression demo.

    python -m edge_ai_compression.demo --quick

Runs entirely on CPU with **synthetic data** — no dataset download, no network.
It takes a small model, benchmarks it, applies pruning + dynamic quantization,
benchmarks the result, and reports the size/latency trade-off plus a JSON and
Markdown summary under ``results/demo/``.

Honesty note: the reported ``synthetic_accuracy`` is measured on randomly
labeled fake images. It is a plumbing/sanity number ONLY and is meaningless as a
quality metric — it is labeled "synthetic" everywhere for exactly this reason.
For real accuracy, run the experiment runner on CIFAR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torchvision

from edge_ai_compression.benchmarking.benchmark_model import benchmark_model
from edge_ai_compression.core.experiment import PruningSection
from edge_ai_compression.core.pipeline import PruningStage, QuantizationStage
from edge_ai_compression.core.registry import ModelRegistry

_DEMO_DIR = Path("results/demo")
_INPUT_SHAPE = (1, 3, 32, 32)


def _synthetic_accuracy(model: torch.nn.Module, samples: int = 128) -> float:
    """Top-1 accuracy on randomly-labeled fake images. Meaningless by design.

    With random labels this hovers around chance (~1/num_classes); it exists only
    to prove the forward path runs end to end, never as a quality signal.
    """
    ds = torchvision.datasets.FakeData(
        size=samples,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=torchvision.transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum())
            total += int(labels.numel())
    return correct / total if total else 0.0


def run_demo(*, quick: bool = True, out_dir: Path = _DEMO_DIR) -> dict[str, Any]:
    """Run the baseline -> compressed comparison and return the report dict."""
    warmup, repeats = (2, 10) if quick else (5, 50)

    baseline = ModelRegistry.create("small_cnn_student", num_classes=10)
    base_report = benchmark_model(
        baseline, _INPUT_SHAPE, warmup=warmup, repeats=repeats, measure_memory=False
    )
    base_report["synthetic_accuracy"] = _synthetic_accuracy(baseline)

    # Compress: global unstructured pruning followed by dynamic quantization.
    compressed = ModelRegistry.create("small_cnn_student", num_classes=10)
    compressed.load_state_dict(baseline.state_dict())
    compressed = PruningStage(PruningSection(enabled=True, amount=0.5)).apply(compressed, None)
    compressed = QuantizationStage(mode="dynamic_linear").apply(compressed, None)
    comp_report = benchmark_model(
        compressed, _INPUT_SHAPE, warmup=warmup, repeats=repeats, measure_memory=False
    )
    comp_report["synthetic_accuracy"] = _synthetic_accuracy(compressed)

    size_ratio = (
        base_report["size_mb"] / comp_report["size_mb"]
        if comp_report["size_mb"] > 0
        else float("nan")
    )
    speedup = (
        base_report["latency_ms"]["mean"] / comp_report["latency_ms"]["mean"]
        if comp_report["latency_ms"]["mean"] > 0
        else float("nan")
    )

    return {
        "mode": "quick" if quick else "full",
        "data": "synthetic (torchvision.FakeData, random labels)",
        "disclaimer": (
            "synthetic_accuracy is measured on randomly labeled fake data and is "
            "NOT a quality metric. Run the CIFAR experiment runner for real accuracy."
        ),
        "baseline": base_report,
        "compressed": comp_report,
        "compression": {
            "size_ratio": round(size_ratio, 4),
            "latency_speedup": round(speedup, 4),
            "param_reduction": round(
                1 - comp_report["num_parameters"] / base_report["num_parameters"], 4
            )
            if base_report["num_parameters"]
            else 0.0,
            "sparsity_after_pruning": comp_report["weight_sparsity"],
        },
    }


def _row(label: str, rep: dict[str, Any]) -> str:
    return (
        f"| {label:<10} | {rep['num_parameters']:>10,} | {rep['size_mb']:>8.3f} | "
        f"{rep['latency_ms']['mean']:>10.3f} | {rep['synthetic_accuracy']:>9.3f} |"
    )


def _to_markdown(report: dict[str, Any]) -> str:
    b, c, comp = report["baseline"], report["compressed"], report["compression"]
    return (
        "# Compression Demo (synthetic data)\n\n"
        f"- Mode: `{report['mode']}`\n"
        f"- Data: {report['data']}\n\n"
        f"> ⚠️ {report['disclaimer']}\n\n"
        "| Model      | Params | Size (MB) | Latency (ms) | Synth. acc |\n"
        "| ---------- | ------ | --------- | ------------ | ---------- |\n"
        f"| baseline   | {b['num_parameters']:,} | {b['size_mb']:.3f} | "
        f"{b['latency_ms']['mean']:.3f} | {b['synthetic_accuracy']:.3f} |\n"
        f"| compressed | {c['num_parameters']:,} | {c['size_mb']:.3f} | "
        f"{c['latency_ms']['mean']:.3f} | {c['synthetic_accuracy']:.3f} |\n\n"
        f"- Size ratio (baseline/compressed): **{comp['size_ratio']}x**\n"
        f"- Latency speedup: **{comp['latency_speedup']}x**\n"
        f"- Parameter reduction: **{comp['param_reduction'] * 100:.1f}%**\n"
        f"- Weight sparsity after pruning: **{comp['sparsity_after_pruning']:.3f}**\n"
    )


def _print_table(report: dict[str, Any]) -> None:
    comp = report["compression"]
    print("\nCompression demo (SYNTHETIC DATA — accuracy is not meaningful)\n")
    print("| Model      |     Params | Size(MB) | Latency(ms) | Synth.acc |")
    print("| ---------- | ---------- | -------- | ----------- | --------- |")
    print(_row("baseline", report["baseline"]))
    print(_row("compressed", report["compressed"]))
    print(
        f"\nsize x{comp['size_ratio']}  |  speedup x{comp['latency_speedup']}  |  "
        f"params -{comp['param_reduction'] * 100:.1f}%  |  "
        f"sparsity {comp['sparsity_after_pruning']:.3f}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quick", action="store_true", help="Fewer benchmark iterations (default).")
    p.add_argument("--full", action="store_true", help="More benchmark iterations.")
    p.add_argument("--out", type=Path, default=_DEMO_DIR, help="Output directory.")
    args = p.parse_args(argv)

    quick = not args.full  # --quick is the default; --full opts into more iters
    report = run_demo(quick=quick, out_dir=args.out)

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "demo_report.json"
    md_path = args.out / "demo_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    _print_table(report)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
