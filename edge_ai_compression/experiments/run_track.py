"""Run a study track end to end: train baseline seeds, then run the track's sweep.

Tracks: resnet18_ptq / vit_s_ptq (Phase 3 PTQ/QAT ladder), resnet18_prune
(Phase 4 pruning + recovery ladder). Used by the Colab/Kaggle notebooks
(``--device cuda``); training is skipped for seeds whose checkpoint exists.
``--smoke`` runs the same code path on synthetic data with tiny settings (CI /
pytest) and writes to ``--results``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edge_ai_compression.experiments.launch_sweep import run_sweep
from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.trainer import run_training
from edge_ai_compression.utils.config_loader import load_yaml
from edge_ai_compression.utils.overrides import apply_overrides

CONFIGS = Path("edge_ai_compression/configs")
TRACKS = {
    "resnet18_ptq": ("train/resnet18_cifar10.yml", "sweeps/ptq_ladder_resnet18_cifar10.yml"),
    "vit_s_ptq": ("train/vit_s_cifar10.yml", "sweeps/ptq_ladder_vit_s_cifar10.yml"),
    "resnet18_prune": ("train/resnet18_cifar10.yml", "sweeps/prune_ladder_resnet18_cifar10.yml"),
}
SMOKE_TRAIN = {
    "dataset": "fake",
    "limit_samples": 32,
    "batch_size": 8,
    "num_workers": 0,
    "epochs": 1,
    "num_checkpoints": 2,
    "log_every_steps": 2,
}
SMOKE_BASE = {
    "dataset": "fake",
    "limit_samples": 32,
    "batch_size": 16,
    "num_workers": 0,
    "compression.quantization.calibration.num_samples": 16,
    "benchmark.iters": 3,
    "benchmark.warmup_iters": 1,
    "benchmark.min_warmup_s": 0.0,
    "benchmark.process_repeats": 1,
    "benchmark.strict_environment": False,
    "experiment_db.diagnostics_max_batches": 1,
}


def _shrink(rung: dict[str, Any]) -> dict[str, Any]:
    """Tiny iteration counts for reconstruction / recovery options inside a rung."""
    comp = rung.get("compression", {})
    q = comp.get("quantization", {})
    for method in ("adaround", "brecq"):
        if method in q:
            q[method] = {**q[method], "iters": 2, "num_samples": 16, "batch_size": 8}
    if "qat" in q:
        q["qat"] = {**q["qat"], "max_steps": 2}
    rec = comp.get("pruning", {}).get("recovery")
    if rec:
        comp["pruning"]["recovery"] = {**rec, "max_steps": 2}
    return rung


def _smoke_sweep(spec: dict[str, Any]) -> dict[str, Any]:
    """First two rungs plus the first rung that trains (reconstruction/QAT/recovery)."""
    seeds, rungs = spec["axes"]
    trains = [r for r in rungs[2:] if any(k in str(r) for k in ("adaround", "recovery", "qat"))]
    return {**spec, "axes": [seeds, [_shrink(r) for r in rungs[:2] + trains[:1]]]}


def run(
    track: str,
    seeds: list[int],
    device: str,
    results: Path,
    models: Path,
    smoke: bool,
    skip_train: bool,
) -> None:
    train_path, sweep_path = TRACKS[track]
    model_prefix = train_path.split("/")[1].removesuffix(".yml")
    for seed in seeds:
        if skip_train or (models / f"{model_prefix}_seed{seed}.pt").is_file():
            continue
        cfg = apply_overrides(
            load_yaml(CONFIGS / train_path),
            {
                "seed": seed,
                "device": device,
                "results_dir": str(results),
                "checkpoint_dir": str(models / "checkpoints"),
                "export_path": str(models / f"{model_prefix}_seed{seed}.pt"),
                **(SMOKE_TRAIN if smoke else {}),
            },
        )
        print(run_training(TrainConfig.from_dict(cfg)))
    spec = load_yaml(CONFIGS / sweep_path)
    spec["axes"][0] = [
        {**entry, "checkpoint_in": str(models / f"{model_prefix}_seed{entry['seed']}.pt")}
        for entry in spec["axes"][0]
        if entry["seed"] in seeds
    ]
    if smoke:
        spec = _smoke_sweep(spec)
    overrides = {
        "device": device,
        "experiment_db.results_dir": str(results),
        "checkpoint_out": str(models / "ladder_last.pt"),
        **(SMOKE_BASE if smoke else {}),
    }
    run_sweep(spec, results / "sweeps", overrides)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--track", choices=sorted(TRACKS), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="never train (checkpoints exist)")
    a = p.parse_args()
    run(a.track, a.seeds, a.device, a.results, a.models, a.smoke, a.skip_train)


if __name__ == "__main__":
    main()
