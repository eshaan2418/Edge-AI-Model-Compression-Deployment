"""Merge an experiment DB produced elsewhere (Colab/Kaggle) into a local one.

Tables (CSV + JSONL) are appended with header checks and de-duplicated by run
id; per-run artifact and training directories are copied if absent. Usage:

    python -m edge_ai_compression.experiment_db.merge --src ~/Downloads/results --dst results
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from edge_ai_compression.experiment_db.writer import SchemaMismatchError

# table file -> id column
TABLES = {
    "experiments.csv": "experiment_id",
    "kernel_benchmarks.csv": "run_id",
    "training_runs.csv": "run_id",
}
JSONL = {
    "experiments.jsonl": "experiment_id",
    "kernel_benchmarks.jsonl": "run_id",
}
DIRS = ("artifacts", "training", "sweeps")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def merge(src: Path, dst: Path) -> dict[str, int]:
    """Returns the number of new rows / entries merged per table or directory."""
    dst.mkdir(parents=True, exist_ok=True)
    added: dict[str, int] = {}
    for name, key in TABLES.items():
        s = src / name
        if not s.is_file():
            continue
        s_fields, s_rows = _read_csv(s)
        d = dst / name
        existing: set[str] = set()
        if d.is_file():
            d_fields, d_rows = _read_csv(d)
            if d_fields != s_fields:
                raise SchemaMismatchError(f"{name}: different columns in {src} and {dst}")
            existing = {r[key] for r in d_rows}
        new = [r for r in s_rows if r[key] not in existing]
        with open(d, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=s_fields)
            if not existing and not (d.is_file() and d.stat().st_size):
                w.writeheader()
            w.writerows(new)
        added[name] = len(new)
    for name, key in JSONL.items():
        s = src / name
        if not s.is_file():
            continue
        d = dst / name
        existing = set()
        if d.is_file():
            existing = {json.loads(line)[key] for line in d.read_text().splitlines() if line}
        lines = [line for line in s.read_text().splitlines() if line]
        new_lines = [line for line in lines if json.loads(line)[key] not in existing]
        with open(d, "a", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in new_lines)
        added[name] = len(new_lines)
    for name in DIRS:
        s = src / name
        if not s.is_dir():
            continue
        count = 0
        for entry in s.iterdir():
            target = dst / name / entry.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, target)
            count += 1
        added[name] = count
    return added


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, default=Path("results"))
    args = p.parse_args()
    for table, n in merge(args.src, args.dst).items():
        print(f"{table}: +{n}")


if __name__ == "__main__":
    main()
