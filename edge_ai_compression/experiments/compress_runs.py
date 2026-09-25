"""Apply a fixed panel of training-free compressions to training-run checkpoints.

For every row of ``training_runs.csv`` (Phase 5), the final checkpoint (or, with
``--all-steps``, every logged checkpoint) is compressed with each panel method
and evaluated; each result is one experiment-DB row joined to the training run
by ``source_run_id`` / ``source_step``. The accuracy drop is the
*compressibility* target of the Phase 6 early-prediction study. Latency is
measured minimally (torch_eager, 1 process): it is not the point of these rows.
Combinations already in the DB are skipped, so re-running resumes.

    python -m edge_ai_compression.experiments.compress_runs --results results --device cuda
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import Any

from edge_ai_compression.core.experiment import ExperimentConfig
from edge_ai_compression.core.runner import ExperimentRunner
from edge_ai_compression.experiment_db.paths import experiments_csv

PANEL: dict[str, dict[str, Any]] = {
    "w8a8": {"quantization": {"enabled": True, "method": "rtn", "weight_bits": 8, "act_bits": 8}},
    "w4a8": {"quantization": {"enabled": True, "method": "rtn", "weight_bits": 4, "act_bits": 8}},
    "w4g32": {
        "quantization": {
            "enabled": True,
            "method": "rtn",
            "weight_bits": 4,
            "act_bits": None,
            "granularity": "per_group",
            "group_size": 32,
        }
    },
    "nm24": {"pruning": {"enabled": True, "mode": "nm", "n": 2, "m": 4}},
    "unstr50": {"pruning": {"enabled": True, "mode": "global_unstructured", "amount": 0.5}},
    "unstr80": {"pruning": {"enabled": True, "mode": "global_unstructured", "amount": 0.8}},
}


def method_tag(method: str) -> tuple[str, str]:
    """(pruning_type, quantization_type) DB tags produced by a panel method."""
    from edge_ai_compression.core.experiment import CompressionConfig

    comp = CompressionConfig.from_dict(PANEL[method])
    prune = comp.pruning.tag if comp.pruning.enabled else "none"
    quant = comp.quantization.tag if comp.quantization.enabled else "none"
    return prune, quant


def experiment_config(
    run: dict[str, str],
    checkpoint: Path,
    method: str,
    results_dir: Path,
    device: str,
    overrides: dict[str, Any],
    scratch: Path,
) -> dict[str, Any]:
    panel = PANEL[method]
    return {
        "model": run["model"],
        "dataset": run["dataset"],
        "data_dir": "data",
        "batch_size": 256,
        "num_workers": 4,
        "device": device,
        "seed": 0,
        "checkpoint_in": str(checkpoint),
        "checkpoint_out": str(scratch / "out.pt"),
        "compression_order_tag": "prune" if "pruning" in panel else "quantize",
        "compression": {
            "pruning": panel.get("pruning", {"enabled": False}),
            "quantization": {
                **panel.get("quantization", {"enabled": False}),
                "calibration": {"method": "minmax", "num_samples": 512},
            },
            "distillation": {"enabled": False},
        },
        "benchmark": {
            "backend": "torch_eager",
            "iters": 1,
            "warmup_iters": 0,
            "min_warmup_s": 0.0,
            "process_repeats": 1,
            "strict_environment": False,
        },
        "experiment_db": {"results_dir": str(results_dir), "diagnostics_max_batches": 10},
        **overrides,
    }


def _done(results_dir: Path) -> set[tuple[str, str, str, str]]:
    path = experiments_csv(results_dir)
    if not path.is_file():
        return set()
    with open(path, newline="") as f:
        return {
            (r["source_run_id"], r["source_step"], r["pruning_type"], r["quantization_type"])
            for r in csv.DictReader(f)
        }


def checkpoints_for(run: dict[str, str], all_steps: bool) -> list[Path]:
    final = Path(run["final_checkpoint"])
    if not all_steps:
        return [final]
    return sorted(final.parent.glob("step_*.pt"))


def compress_runs(
    results_dir: Path,
    *,
    methods: list[str],
    device: str = "cpu",
    all_steps: bool = False,
    overrides: dict[str, Any] | None = None,
) -> int:
    """Run every missing (checkpoint, method) combination. Returns the number of new rows."""
    import torch

    with open(results_dir / "training_runs.csv", newline="") as f:
        runs = list(csv.DictReader(f))
    done = _done(results_dir)
    new = 0
    with tempfile.TemporaryDirectory() as tmp:
        for run in runs:
            for ckpt in checkpoints_for(run, all_steps):
                step = str(torch.load(ckpt, map_location="cpu", weights_only=False)["step"])
                for method in methods:
                    if (run["run_id"], step, *method_tag(method)) in done:
                        continue
                    cfg = experiment_config(
                        run, ckpt, method, results_dir, device, overrides or {}, Path(tmp)
                    )
                    ExperimentRunner(ExperimentConfig.from_dict(cfg)).run()
                    new += 1
    return new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--methods", nargs="+", default=list(PANEL), choices=list(PANEL))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--all-steps", action="store_true", help="every logged checkpoint, not just final"
    )
    a = p.parse_args()
    n = compress_runs(a.results, methods=a.methods, device=a.device, all_steps=a.all_steps)
    print(f"{n} new rows")


if __name__ == "__main__":
    main()
