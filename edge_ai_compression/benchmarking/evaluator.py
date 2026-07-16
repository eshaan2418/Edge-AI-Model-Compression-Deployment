from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.benchmarking.energy_estimator import estimate_energy_proxy
from edge_ai_compression.benchmarking.latency_profiler import profile_model_detailed
from edge_ai_compression.benchmarking.memory_profiler import estimate_peak_rss_mib


@dataclass
class BenchmarkReport:
    accuracy: float
    latency_ms_mean: float
    latency_ms_p99: float
    size_mb: float
    peak_ram_mib: float
    energy_proxy: float
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_p99": self.latency_ms_p99,
            "size_mb": self.size_mb,
            "peak_ram_mib": self.peak_ram_mib,
            "energy_proxy": self.energy_proxy,
            **self.extras,
        }


def _state_dict_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def _accuracy_on_loader(
    model: nn.Module, test_loader: DataLoader, device: str
) -> tuple[float, float]:
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
    def __init__(
        self,
        device: str = "cpu",
        latency_repeats: int = 100,
        latency_warmup: int = 3,
    ) -> None:
        self.device = device
        self.latency_repeats = latency_repeats
        self.latency_warmup = latency_warmup

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> BenchmarkReport:
        accuracy, val_loss = _accuracy_on_loader(model, test_loader, self.device)
        sample = next(iter(test_loader))[0][:1].to(self.device)
        det = profile_model_detailed(
            model.to(self.device),
            sample,
            warmup=self.latency_warmup,
            repeats=self.latency_repeats,
        )
        trace = det["trace_ms"]
        assert isinstance(trace, list)

        def forward_once() -> None:
            with torch.no_grad():
                _ = model(sample)

        peak_ram = estimate_peak_rss_mib(forward_once)
        size_mb = _state_dict_size_mb(model)
        energy = estimate_energy_proxy(float(det["mean"]), size_mb)

        latency_stats = {k: float(v) for k, v in det.items() if k != "trace_ms"}

        return BenchmarkReport(
            accuracy=float(accuracy),
            latency_ms_mean=float(det["mean"]),
            latency_ms_p99=float(det["p99"]),
            size_mb=float(size_mb),
            peak_ram_mib=float(peak_ram),
            energy_proxy=energy.score,
            extras={
                "val_loss_approx": val_loss,
                "latency": latency_stats,
                "latency_trace_ms": trace,
                "cold_start_ms": det["cold_start_ms"],
                "throughput_ips": det["throughput_ips"],
            },
        )


class ResultLogger:
    @staticmethod
    def log_md(path: str, name: str, row: dict[str, Any]) -> None:
        header = (
            "| Model | Size (MB) | Latency mean (ms) | p99 (ms) | "
            "RAM (MiB) | Accuracy | Energy proxy |\n"
        )
        sep = "|:---|---:|---:|---:|---:|---:|---:|\n"
        line = (
            f"| {name} | {row['size_mb']:.2f} | {row['latency_ms_mean']:.2f} | "
            f"{row['latency_ms_p99']:.2f} | {row['peak_ram_mib']:.2f} | "
            f"{row['accuracy'] * 100:.2f}% | "
            f"{row['energy_proxy']:.2f} |\n"
        )
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(header + sep + line)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
