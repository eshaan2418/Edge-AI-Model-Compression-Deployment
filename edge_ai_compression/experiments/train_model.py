"""Train a model from a YAML TrainConfig; the run is logged to the experiment DB."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.trainer import run_training
from edge_ai_compression.utils.config_loader import load_yaml
from edge_ai_compression.utils.overrides import apply_overrides, parse_overrides


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (value parsed as YAML), e.g. --set seed=1",
    )
    args = p.parse_args()
    cfg = apply_overrides(load_yaml(args.config), parse_overrides(args.set))
    result = run_training(TrainConfig.from_dict(cfg))
    print(
        f"run {result.run_id}: test accuracy {result.test_accuracy:.4f} after "
        f"{result.steps} steps; final checkpoint {result.checkpoints[-1]}"
    )


if __name__ == "__main__":
    main()
