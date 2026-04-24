from __future__ import annotations

import argparse
from pathlib import Path

from edge_ai_compression.core.runner import ExperimentRunner, load_experiment_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = load_experiment_config(args.config)
    res = ExperimentRunner(cfg).run()
    print(res.to_dict())


if __name__ == "__main__":
    main()
