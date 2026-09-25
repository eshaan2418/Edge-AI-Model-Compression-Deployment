"""Phase 3 end to end: train baseline seeds, then run the PTQ/QAT ladder sweeps.

Used by notebooks/phase3_ladder.ipynb on Colab/Kaggle (``--device cuda``).
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
    "resnet18": ("train/resnet18_cifar10.yml", "sweeps/ptq_ladder_resnet18_cifar10.yml"),
    "vit_s": ("train/vit_s_cifar10.yml", "sweeps/ptq_ladder_vit_s_cifar10.yml"),
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


def _smoke_sweep(spec: dict[str, Any]) -> dict[str, Any]:
    """First two rungs plus the first reconstruction rung, with 2 iterations."""
    seeds, rungs = spec["axes"]
    keep = rungs[:2] + [r for r in rungs if "adaround" in str(r)][:1]
    for rung in keep:
        q = rung.get("compression", {}).get("quantization", {})
        if "adaround" in q:
            q["adaround"] = {**q["adaround"], "iters": 2, "num_samples": 16, "batch_size": 8}
    return {**spec, "axes": [seeds, keep]}


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
        if skip_train:
            break
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
    p.add_argument("--track", choices=sorted(TRACKS), default="resnet18")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="checkpoints already exist")
    a = p.parse_args()
    run(a.track, a.seeds, a.device, a.results, a.models, a.smoke, a.skip_train)


if __name__ == "__main__":
    main()
