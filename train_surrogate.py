#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from edge_ai_compression.surrogate.train import train_surrogates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/experiments.csv"))
    p.add_argument("--out", type=Path, default=Path("results/surrogates"))
    args = p.parse_args()
    meta = train_surrogates(args.results, args.out)
    print(meta)


if __name__ == "__main__":
    main()
