from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.fingerprint import check_environment, collect
from edge_ai_compression.benchmarking.isolation import run_isolated
from edge_ai_compression.benchmarking.memory import MIB
from edge_ai_compression.benchmarking.report import BenchmarkReport
from edge_ai_compression.inference.backends import Runner, get_backend, run_batched


def accuracy_on_loader(
    model: nn.Module, test_loader: DataLoader, device: str
) -> tuple[float, float]:
    """Top-1 accuracy and mean per-batch cross-entropy."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss_sum += criterion(logits, targets).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    acc = correct / max(total, 1)
    return float(acc), loss_sum / max(len(test_loader), 1)


def accuracy_with_runner(run: Runner, test_loader: DataLoader, batch: int) -> tuple[float, float]:
    """Accuracy and mean per-batch cross-entropy using an exported backend runner."""
    correct = total = 0
    loss_sum = 0.0
    for images, targets in test_loader:
        logits = run_batched(run, images.numpy(), batch)
        t = torch.from_numpy(logits)
        loss_sum += nn.functional.cross_entropy(t, targets).item()
        correct += int((np.argmax(logits, axis=1) == targets.numpy()).sum())
        total += targets.numel()
    return correct / max(total, 1), loss_sum / max(len(test_loader), 1)


class Evaluator:
    """Accuracy and latency/memory for ``config.backend``.

    The model is exported once for the backend; ``size_mb`` is the artifact size
    on disk. Latency and memory are always measured on CPU in fresh processes.
    Accuracy uses the same backend (a quantized engine's accuracy differs from
    eager); only ``torch_eager`` evaluates on ``device``.
    """

    def __init__(self, device: str, config: BenchmarkConfig) -> None:
        self.device = device
        self.config = config

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> BenchmarkReport:
        cfg = self.config
        fingerprint = collect()
        problems = check_environment(fingerprint, strict=cfg.strict_environment)
        sample_shape = tuple(next(iter(test_loader))[0].shape[1:])
        input_shape = (cfg.batch_size, *sample_shape)
        backend = get_backend(cfg.backend)
        with tempfile.TemporaryDirectory() as tmp:
            path = backend.export(model, input_shape, Path(tmp))
            size_mb = path.stat().st_size / MIB
            if cfg.backend == "torch_eager":
                accuracy, val_loss = accuracy_on_loader(model, test_loader, self.device)
            else:
                run = backend.load(path, cfg.num_threads)
                accuracy, val_loss = accuracy_with_runner(run, test_loader, cfg.batch_size)
            runs = [run_isolated(path, input_shape, cfg) for _ in range(cfg.process_repeats)]
        return BenchmarkReport.from_runs(
            runs,
            accuracy=accuracy,
            val_loss=val_loss,
            input_shape=input_shape,
            config=cfg,
            size_mb=size_mb,
            fingerprint=fingerprint,
            environment_problems=problems,
        )
