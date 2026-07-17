"""Validate an experiment YAML config so runs fail fast with helpful errors.

    python -m edge_ai_compression.config.validate \
        edge_ai_compression/configs/experiments/smoke_cpu.yml [--strict]

Validation targets the *actual* flat config schema consumed by
``core.runner`` / ``ExperimentConfig.from_dict`` — top-level ``model`` and
``dataset`` are strings (not nested ``model.name``). ``--strict`` additionally
rejects unknown keys, catching typos like ``bach_size``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from edge_ai_compression.core.registry import ModelRegistry

# Known keys, matching ExperimentConfig.from_dict.
_TOP_LEVEL_KEYS = {
    "model",
    "dataset",
    "data_dir",
    "batch_size",
    "num_workers",
    "device",
    "seed",
    "limit_samples",
    "checkpoint_in",
    "checkpoint_out",
    "compression",
    "compression_order",
    "compression_order_tag",
    "hardware_profile",
    "benchmark",
    "train",
    "experiment_db",
}
_COMPRESSION_KEYS = {"pruning", "quantization", "distillation"}
_PRUNING_KEYS = {"enabled", "amount", "mode", "scorer"}
_QUANT_KEYS = {"enabled", "mode"}
_DISTILL_KEYS = {
    "enabled",
    "teacher_ckpt",
    "teacher_model",
    "temperature",
    "alpha",
    "epochs",
    "lr",
}

_KNOWN_DATASETS = {
    "cifar10",
    "cifar100",
    "tiny_imagenet",
    "tiny-imagenet-200",
    "fake",
    "synthetic",
    "debug",
    "random",
}
_PRUNING_MODES = {"global_unstructured", "layerwise_adaptive"}
_PRUNING_SCORERS = {"magnitude", "gradient", "activation", "ablation"}
_QUANT_MODES = {"dynamic_linear"}


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


def _check_type(issues: list[ValidationIssue], d: dict, key: str, types: tuple, field: str) -> None:
    if key in d and d[key] is not None and not isinstance(d[key], types):
        names = "/".join(t.__name__ for t in types)
        got = type(d[key]).__name__
        issues.append(ValidationIssue("error", field, f"must be {names}, got {got}"))


def validate_config(config: dict[str, Any], *, strict: bool = False) -> list[ValidationIssue]:
    """Return a list of validation issues (empty == valid). Never raises."""
    issues: list[ValidationIssue] = []

    if not isinstance(config, dict):
        return [ValidationIssue("error", "<root>", "config must be a YAML mapping")]

    # --- required: model ---
    model = config.get("model")
    if not model:
        issues.append(ValidationIssue("error", "model", "required field is missing or empty"))
    elif not isinstance(model, str):
        issues.append(ValidationIssue("error", "model", "must be a string model name"))
    elif model not in ModelRegistry.available():
        issues.append(
            ValidationIssue(
                "error",
                "model",
                f"unknown model '{model}'. Available: {ModelRegistry.available()}",
            )
        )

    # --- required: dataset ---
    dataset = config.get("dataset")
    if not dataset:
        issues.append(ValidationIssue("error", "dataset", "required field is missing or empty"))
    elif not isinstance(dataset, str):
        issues.append(ValidationIssue("error", "dataset", "must be a string dataset name"))
    elif dataset.lower() not in _KNOWN_DATASETS:
        issues.append(
            ValidationIssue(
                "error",
                "dataset",
                f"unknown dataset '{dataset}'. Known: {sorted(_KNOWN_DATASETS)}",
            )
        )

    # --- typed optional scalars ---
    _check_type(issues, config, "batch_size", (int,), "batch_size")
    _check_type(issues, config, "num_workers", (int,), "num_workers")
    _check_type(issues, config, "seed", (int,), "seed")
    _check_type(issues, config, "limit_samples", (int,), "limit_samples")
    _check_type(issues, config, "device", (str,), "device")
    if isinstance(config.get("batch_size"), int) and config["batch_size"] <= 0:
        issues.append(ValidationIssue("error", "batch_size", "must be > 0"))

    # --- compression section ---
    comp = config.get("compression")
    if comp is not None:
        if not isinstance(comp, dict):
            issues.append(ValidationIssue("error", "compression", "must be a mapping"))
        else:
            _validate_compression(issues, comp, strict=strict)

    # --- strict: unknown top-level keys ---
    if strict:
        for key in config:
            if key not in _TOP_LEVEL_KEYS:
                issues.append(ValidationIssue("error", key, "unknown top-level key (strict mode)"))

    return issues


def _validate_compression(issues: list[ValidationIssue], comp: dict, *, strict: bool) -> None:
    pruning = comp.get("pruning")
    if isinstance(pruning, dict):
        mode = pruning.get("mode")
        if mode is not None and mode not in _PRUNING_MODES:
            issues.append(
                ValidationIssue(
                    "error", "compression.pruning.mode", f"must be one of {sorted(_PRUNING_MODES)}"
                )
            )
        scorer = pruning.get("scorer")
        if scorer is not None and scorer not in _PRUNING_SCORERS:
            issues.append(
                ValidationIssue(
                    "error",
                    "compression.pruning.scorer",
                    f"must be one of {sorted(_PRUNING_SCORERS)}",
                )
            )
        amount = pruning.get("amount")
        if isinstance(amount, (int, float)) and not (0.0 <= amount < 1.0):
            issues.append(
                ValidationIssue("error", "compression.pruning.amount", "must be in [0.0, 1.0)")
            )
        if strict:
            for k in pruning:
                if k not in _PRUNING_KEYS:
                    issues.append(
                        ValidationIssue("error", f"compression.pruning.{k}", "unknown key (strict)")
                    )

    quant = comp.get("quantization")
    if isinstance(quant, dict):
        mode = quant.get("mode")
        if mode is not None and mode not in _QUANT_MODES:
            issues.append(
                ValidationIssue(
                    "error",
                    "compression.quantization.mode",
                    f"must be one of {sorted(_QUANT_MODES)}",
                )
            )
        if strict:
            for k in quant:
                if k not in _QUANT_KEYS:
                    issues.append(
                        ValidationIssue(
                            "error", f"compression.quantization.{k}", "unknown key (strict)"
                        )
                    )

    distill = comp.get("distillation")
    if isinstance(distill, dict) and strict:
        for k in distill:
            if k not in _DISTILL_KEYS:
                issues.append(
                    ValidationIssue(
                        "error", f"compression.distillation.{k}", "unknown key (strict)"
                    )
                )

    if strict:
        for k in comp:
            if k not in _COMPRESSION_KEYS:
                issues.append(ValidationIssue("error", f"compression.{k}", "unknown key (strict)"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.config.validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to an experiment YAML config.")
    p.add_argument("--strict", action="store_true", help="Reject unknown keys.")
    args = p.parse_args(argv)

    if not args.config.is_file():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 2

    try:
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"Invalid YAML in {args.config}: {exc}", file=sys.stderr)
        return 2

    issues = validate_config(loaded or {}, strict=args.strict)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for issue in issues:
        print(issue)

    if errors:
        print(f"\n✗ {args.config}: {len(errors)} error(s), {len(warnings)} warning(s).")
        return 1
    print(f"✓ {args.config} is valid ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
