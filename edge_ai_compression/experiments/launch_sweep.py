from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from edge_ai_compression.core.experiment import ExperimentConfig
from edge_ai_compression.core.runner import ExperimentRunner
from edge_ai_compression.utils.config_loader import merge_dict


def _run_variant(base_path: Path, variant: dict[str, object], sweep_id: str) -> dict[str, object]:
    cfg_dict = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    merged = merge_dict(cfg_dict, variant)
    cfg = ExperimentConfig.from_dict(merged)
    res = ExperimentRunner(cfg).run()
    return {"variant": variant, "result": res.to_dict(), "sweep_id": sweep_id}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch compression sweep (sequential runner; Phase 14)"
    )
    p.add_argument(
        "--config", type=Path, required=True, help="Sweep YAML with base_config and variants"
    )
    args = p.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = Path(spec["base_config"])
    sweep_id = str(spec.get("sweep_id", "sweep"))
    variants = list(spec.get("variants", []))
    state_path = Path("results") / "sweeps" / f"{sweep_id}_state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    for v in variants:
        row = _run_variant(base, v, sweep_id)
        with open(state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
