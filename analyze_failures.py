#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    args = p.parse_args()
    root = Path("results") / "artifacts" / args.experiment_id
    fc = root / "failure_cases.json"
    m = root / "metrics.json"
    if not m.is_file():
        print(f"Missing {m}")
        return
    metrics = json.loads(m.read_text(encoding="utf-8"))
    print(json.dumps(metrics.get("diagnostics", metrics), indent=2)[:8000])
    if fc.is_file():
        print("\n--- failure_cases.json (truncated) ---\n")
        data = json.loads(fc.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2)[:4000])


if __name__ == "__main__":
    main()
