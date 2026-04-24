from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Show sweep progress / completed variants (Phase 14)")
    p.add_argument("--sweep-id", required=True)
    p.add_argument("--tail", type=int, default=20)
    args = p.parse_args()
    path = Path("results") / "sweeps" / f"{args.sweep_id}_state.jsonl"
    if not path.is_file():
        print(f"No state file at {path}")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines[-args.tail :]:
        print(line)


if __name__ == "__main__":
    main()
