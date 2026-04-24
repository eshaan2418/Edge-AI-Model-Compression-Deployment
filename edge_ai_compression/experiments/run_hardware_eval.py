from __future__ import annotations

import argparse

import torch

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.hardware.cpu_runner import CpuRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18_cifar")
    args = p.parse_args()
    m = ModelRegistry.create(args.model)
    x = torch.randn(1, 3, 32, 32)
    stats = CpuRunner().run_inference(m, x)
    print(stats)


if __name__ == "__main__":
    main()
