from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from edge_ai_compression.core.experiment import CompressionConfig, PruningSection

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CompressionStage(ABC):
    @abstractmethod
    def apply(self, model: nn.Module, data: Any) -> nn.Module:
        raise NotImplementedError


class PruningStage(CompressionStage):
    def __init__(self, section: PruningSection) -> None:
        self.section = section
        self.device: str = "cpu"
        self.baseline_accuracy: float | None = None

    def apply(self, model: nn.Module, data: Any) -> nn.Module:
        if self.section.mode == "layerwise_adaptive":
            from edge_ai_compression.compression.pruning.layerwise.apply import (
                apply_layerwise_l1_pruning,
                compute_layerwise_sparsities,
            )

            assert data is not None, "layerwise pruning requires train DataLoader"
            sparsities = compute_layerwise_sparsities(
                model,
                self.section.scorer,
                self.section.amount,
                data,
                self.device,
                self.baseline_accuracy,
            )
            return apply_layerwise_l1_pruning(model, sparsities)

        from edge_ai_compression.compression.pruning.global_unstructured import (
            apply_global_unstructured_pruning,
            remove_pruning_reparametrization,
        )

        model = apply_global_unstructured_pruning(model, amount=self.section.amount)
        return remove_pruning_reparametrization(model)


class QuantizationStage(CompressionStage):
    def __init__(self, mode: str = "dynamic_linear") -> None:
        from edge_ai_compression.compression.quantization.dynamic import dynamic_quantize_linear_layers

        self.mode = mode
        self._quantize = dynamic_quantize_linear_layers

    def apply(self, model: nn.Module, data: Any) -> nn.Module:
        if self.mode != "dynamic_linear":
            raise ValueError(f"Unsupported quantization mode: {self.mode}")
        return self._quantize(model)


class DistillationStage(CompressionStage):
    def __init__(
        self,
        teacher_ckpt: str,
        teacher_model: str,
        temperature: float,
        alpha: float,
        epochs: int,
        lr: float,
        data_dir: str,
        batch_size: int,
        device: str,
    ) -> None:
        self.teacher_ckpt = teacher_ckpt
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device

    def apply(self, model: nn.Module, data: Any) -> nn.Module:
        from edge_ai_compression.compression.distillation.trainer import distill_student_inplace

        return distill_student_inplace(
            student=model,
            teacher_model=self.teacher_model,
            teacher_ckpt=self.teacher_ckpt,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            device=self.device,
            epochs=self.epochs,
            lr=self.lr,
            temperature=self.temperature,
            alpha=self.alpha,
        )


def build_stages_for_order(
    order: list[str],
    cfg: CompressionConfig,
) -> list[CompressionStage]:
    stages: list[CompressionStage] = []
    for name in order:
        if name == "distill" and cfg.distillation.enabled:
            stages.append(
                DistillationStage(
                    teacher_ckpt=cfg.distillation.teacher_ckpt,
                    teacher_model=cfg.distillation.teacher_model,
                    temperature=cfg.distillation.temperature,
                    alpha=cfg.distillation.alpha,
                    epochs=cfg.distillation.epochs,
                    lr=cfg.distillation.lr,
                    data_dir="",
                    batch_size=128,
                    device="cpu",
                )
            )
        elif name == "prune" and cfg.pruning.enabled:
            stages.append(PruningStage(cfg.pruning))
        elif name == "quantize" and cfg.quantization.enabled:
            stages.append(QuantizationStage(mode=cfg.quantization.mode))
    return stages


class CompressionPipeline:
    def __init__(self, config: CompressionConfig, order: list[str] | None = None) -> None:
        self.config = config
        self.order = order or ["distill", "prune", "quantize"]
        self.stages = build_stages_for_order(self.order, config)

    def run(
        self,
        model: nn.Module,
        data: "DataLoader | Any",
        *,
        data_dir: str = "data",
        batch_size: int = 128,
        device: str = "cpu",
        baseline_accuracy: float | None = None,
    ) -> nn.Module:
        for stage in self.stages:
            if isinstance(stage, DistillationStage):
                stage.data_dir = data_dir
                stage.batch_size = batch_size
                stage.device = device
            if isinstance(stage, PruningStage):
                stage.device = device
                stage.baseline_accuracy = baseline_accuracy
            model = stage.apply(model, data)
        return model
