"""Generate a Markdown model card for a compressed/exported artifact.

    python -m edge_ai_compression.reporting.model_card \
        --metrics results/demo/report.json \
        --profile raspberry_pi \
        --out reports/model_card.md

The card is deliberately honest: if metrics came from synthetic data it says so
and does not present them as quality evidence, and it never claims production
readiness unless real (non-synthetic) benchmark metrics are supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edge_ai_compression.utils.metrics_io import extract_metrics, load_json


def _fmt(val: Any, spec: str = "") -> str:
    if val is None:
        return "_not measured_"
    if spec and isinstance(val, (int, float)):
        return format(val, spec)
    return str(val)


def build_model_card(
    metrics: dict[str, Any] | None,
    *,
    model_name: str,
    dataset: str,
    profile: str | None = None,
    artifacts: list[str] | None = None,
    repro_command: str | None = None,
) -> str:
    """Return a Markdown model card. All inputs are optional/best-effort."""
    canonical = extract_metrics(metrics) if metrics else {}
    synthetic = canonical.get("accuracy_is_synthetic", False)
    has_real_metrics = bool(canonical) and not synthetic

    lines: list[str] = [f"# Model Card — {model_name}", ""]

    # --- Intended use ---
    lines += [
        "## Intended use",
        "",
        "Edge/on-device image classification research. This artifact is a "
        "compressed convolutional network intended for benchmarking the "
        "accuracy/latency/size trade-off of compression techniques.",
        "",
    ]

    # --- Data ---
    lines += ["## Training & evaluation data", ""]
    if synthetic or dataset.lower() in {"fake", "synthetic", "debug", "random"}:
        lines += [
            f"- Dataset: `{dataset}` — **SYNTHETIC (randomly labeled) data**.",
            "- ⚠️ Any accuracy below is a plumbing/sanity number, **not** a measure "
            "of real model quality. Re-run on CIFAR/Tiny-ImageNet for real accuracy.",
        ]
    else:
        lines += [f"- Dataset: `{dataset}`."]
    lines.append("")

    # --- Compression methods ---
    lines += [
        "## Compression methods",
        "",
        "Applied via `edge_ai_compression` pipeline stages (see the source config "
        "for exact settings): pruning, quantization, and/or distillation.",
        "",
    ]

    # --- Metrics ---
    lines += ["## Metrics", ""]
    if canonical:
        acc_label = "Accuracy (SYNTHETIC)" if synthetic else "Accuracy"
        lines += [
            "| Metric | Value |",
            "| ------ | ----- |",
            f"| {acc_label} | {_fmt(canonical.get('accuracy'), '.4f')} |",
            f"| Latency mean (ms) | {_fmt(canonical.get('latency_ms'), '.3f')} |",
            f"| Size (MB) | {_fmt(canonical.get('size_mb'), '.3f')} |",
            f"| Peak RAM (MiB) | {_fmt(canonical.get('ram_mb'), '.1f')} |",
            f"| Parameters | {_fmt(canonical.get('num_parameters'), ',d')} |",
        ]
    else:
        lines.append("_No metrics supplied._")
    lines.append("")

    # --- Hardware constraints ---
    lines += ["## Hardware constraints", ""]
    if profile:
        from edge_ai_compression.hardware.profiles import get_profile
        from edge_ai_compression.hardware.scoring import score_candidate

        p = get_profile(profile)
        lines += [
            f"- Target profile: **{p.name}** (planning budget, not a measured device ceiling).",
            f"- Budget: latency ≤ {p.max_latency_ms:g} ms, size ≤ {p.max_size_mb:g} MB, "
            f"RAM ≤ {p.max_ram_mb:g} MB.",
            f"- Preferred export format: `{p.preferred_export_format}`.",
        ]
        if canonical:
            result = score_candidate(
                {
                    "accuracy": canonical.get("accuracy"),
                    "latency_ms": canonical.get("latency_ms"),
                    "size_mb": canonical.get("size_mb"),
                    "ram_mb": canonical.get("ram_mb"),
                },
                p,
            )
            verdict = "FEASIBLE" if result.feasible else "INFEASIBLE"
            lines.append(f"- Feasibility on {p.name}: **{verdict}** (score {result.score:.4f}).")
            for v in result.violations:
                lines.append(f"  - Violation: {v}")
    else:
        lines.append("_No target hardware profile specified._")
    lines.append("")

    # --- Limitations ---
    lines += [
        "## Limitations",
        "",
        "- Evaluated on a narrow benchmark; generalization is unverified.",
        "- Latency/size are measured on the benchmarking host, not the target device.",
        "- Hardware profiles are documented budgets, not physical-device measurements.",
    ]
    if synthetic:
        lines.append(
            "- **Accuracy figures come from synthetic data and are not valid quality evidence.**"
        )
    lines.append("")

    # --- Ethics / safety ---
    lines += [
        "## Ethical & safety considerations",
        "",
        "- Not validated for safety-critical or high-stakes decision-making.",
        "- Compression can change error distribution across classes/subgroups; audit "
        "per-class behavior before any real deployment.",
        "",
    ]

    # --- Production readiness statement (honest) ---
    lines += ["## Production readiness", ""]
    if has_real_metrics:
        lines.append(
            "Real benchmark metrics were supplied. This card documents measured "
            "performance, but production use still requires on-device validation."
        )
    else:
        lines.append(
            "**Not production-ready.** No real (non-synthetic) benchmark metrics were "
            "supplied, so no quality claims are made."
        )
    lines.append("")

    # --- Reproducibility ---
    lines += ["## Reproducibility", ""]
    lines.append("```bash")
    lines.append(repro_command or "python -m edge_ai_compression.demo --quick")
    lines.append("```")
    lines.append("")

    # --- Artifacts ---
    lines += ["## Artifacts", ""]
    if artifacts:
        lines += [f"- `{a}`" for a in artifacts]
    else:
        lines.append("_No artifact paths provided._")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.reporting.model_card",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metrics", type=Path, default=None, help="Metrics/report JSON.")
    p.add_argument("--model-name", default="compressed-model", help="Model name for the card.")
    p.add_argument("--dataset", default="fake", help="Dataset the metrics came from.")
    p.add_argument("--profile", default=None, help="Target hardware profile.")
    p.add_argument("--artifact", action="append", default=None, help="Artifact path (repeatable).")
    p.add_argument("--repro-command", default=None, help="Command to reproduce the run.")
    p.add_argument("--out", type=Path, default=Path("reports/model_card.md"))
    args = p.parse_args(argv)

    metrics = None
    if args.metrics is not None:
        if not args.metrics.is_file():
            print(f"Metrics file not found: {args.metrics}")
            return 2
        metrics = load_json(args.metrics)

    card = build_model_card(
        metrics,
        model_name=args.model_name,
        dataset=args.dataset,
        profile=args.profile,
        artifacts=args.artifact,
        repro_command=args.repro_command,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
