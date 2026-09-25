from __future__ import annotations

import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.config import BenchmarkConfig
from edge_ai_compression.benchmarking.fingerprint import check_environment, collect
from edge_ai_compression.benchmarking.isolation import benchmark_model
from edge_ai_compression.benchmarking.report import BenchmarkReport


def state_dict_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


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


class Evaluator:
    """Accuracy on ``device``; latency and memory always on CPU in fresh processes."""

    def __init__(self, device: str, config: BenchmarkConfig) -> None:
        self.device = device
        self.config = config

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> BenchmarkReport:
        fingerprint = collect()
        problems = check_environment(fingerprint, strict=self.config.strict_environment)
        accuracy, val_loss = accuracy_on_loader(model, test_loader, self.device)
        sample_shape = tuple(next(iter(test_loader))[0].shape[1:])
        input_shape = (self.config.batch_size, *sample_shape)
        runs = benchmark_model(model, input_shape, self.config)
        return BenchmarkReport.from_runs(
            runs,
            accuracy=accuracy,
            val_loss=val_loss,
            input_shape=input_shape,
            config=self.config,
            size_mb=state_dict_size_mb(model),
            fingerprint=fingerprint,
            environment_problems=problems,
        )
