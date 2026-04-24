from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path("results")
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
EXPERIMENTS_JSONL = RESULTS_DIR / "experiments.jsonl"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"


def artifact_dir(experiment_id: str) -> Path:
    return ARTIFACTS_DIR / experiment_id
