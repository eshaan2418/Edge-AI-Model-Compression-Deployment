"""Run a sweep of experiment variants sequentially, resuming where it left off.

Sweep YAML:
  sweep_id: name (state file: results/sweeps/<sweep_id>_state.jsonl)
  kind: experiment (default; ExperimentRunner) | train (run_training on a TrainConfig)
  base_config: experiment (or training) YAML every variant is merged into
  variants: [ {override...}, ... ]            # explicit list, and/or
  axes: [ [ {override}, ... ], [ ... ] ]      # cartesian product of override lists

Each variant is deep-merged into the base config; top-level string values may
use {key} templates filled from the merged config (e.g. "models/{model}_s{seed}.pt").
Variants already recorded in the state file are skipped, so re-running the same
command resumes an interrupted sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from edge_ai_compression.core.experiment import ExperimentConfig
from edge_ai_compression.core.runner import ExperimentRunner
from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.trainer import run_training
from edge_ai_compression.utils.config_loader import merge_dict
from edge_ai_compression.utils.overrides import apply_overrides, parse_overrides


def expand_variants(spec: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [dict(v) for v in spec.get("variants", [])]
    axes = spec.get("axes")
    if axes:
        for combo in itertools.product(*axes):
            merged: dict[str, Any] = {}
            for part in combo:
                merged = merge_dict(merged, part)
            variants.append(merged)
    if not variants:
        raise ValueError("sweep needs `variants` and/or `axes`")
    return variants


def variant_key(variant: dict[str, Any]) -> str:
    return json.dumps(variant, sort_keys=True, default=str)


def run_sweep(
    spec: dict[str, Any],
    state_dir: Path = Path("results") / "sweeps",
    base_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run every variant not yet in the state file. ``base_overrides`` (dotted keys)
    are applied to the base config before merging variants, e.g. {"device": "cuda"}."""
    base = yaml.safe_load(Path(spec["base_config"]).read_text(encoding="utf-8"))
    base = apply_overrides(base, base_overrides or {})
    sweep_id = str(spec.get("sweep_id", "sweep"))
    kind = str(spec.get("kind", "experiment"))
    if kind not in ("experiment", "train"):
        raise ValueError("sweep kind must be 'experiment' or 'train'")
    variants = expand_variants(spec)
    state_path = state_dir / f"{sweep_id}_state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if state_path.is_file():
        done = {
            variant_key(json.loads(line)["variant"]) for line in state_path.read_text().splitlines()
        }
    for i, variant in enumerate(variants, 1):
        if variant_key(variant) in done:
            print(f"[{i}/{len(variants)}] skip (done): {variant}")
            continue
        merged = fill_templates(merge_dict(base, variant))
        if kind == "train":
            train = run_training(TrainConfig.from_dict(merged))
            result, run_id = (
                {
                    "run_id": train.run_id,
                    "test_accuracy": train.test_accuracy,
                    "final_checkpoint": train.checkpoints[-1],
                },
                train.run_id,
            )
        else:
            exp = ExperimentRunner(ExperimentConfig.from_dict(merged)).run()
            result, run_id = exp.to_dict(), exp.extras["experiment_id"]
        with open(state_path, "a", encoding="utf-8") as f:
            row = {"variant": variant, "result": result, "sweep_id": sweep_id, "kind": kind}
            f.write(json.dumps(row, default=str) + "\n")
        print(f"[{i}/{len(variants)}] done: {variant} -> {run_id}")
    return state_path


def fill_templates(cfg: dict[str, Any]) -> dict[str, Any]:
    """Format top-level string values like "models/{model}_s{seed}.pt" with the
    config's own top-level scalar values (so grid axes yield distinct paths)."""
    scalars = {k: v for k, v in cfg.items() if isinstance(v, str | int | float)}
    return {
        k: v.format(**scalars) if isinstance(v, str) and "{" in v else v for k, v in cfg.items()
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="sweep YAML")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a base-config key (dotted, YAML value), e.g. --set device=cuda",
    )
    args = p.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_sweep(spec, base_overrides=parse_overrides(args.set))


if __name__ == "__main__":
    main()
