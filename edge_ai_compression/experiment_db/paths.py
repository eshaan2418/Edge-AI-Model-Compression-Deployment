from __future__ import annotations

from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")


def experiments_csv(results_dir: Path) -> Path:
    return results_dir / "experiments.csv"


def experiments_jsonl(results_dir: Path) -> Path:
    return results_dir / "experiments.jsonl"


def artifact_dir(results_dir: Path, experiment_id: str) -> Path:
    return results_dir / "artifacts" / experiment_id


def kernel_benchmarks_csv(results_dir: Path) -> Path:
    return results_dir / "kernel_benchmarks.csv"


def kernel_benchmarks_jsonl(results_dir: Path) -> Path:
    return results_dir / "kernel_benchmarks.jsonl"
