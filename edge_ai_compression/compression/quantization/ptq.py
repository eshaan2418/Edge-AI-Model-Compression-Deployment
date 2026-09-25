"""Run one rung of the PTQ/QAT ladder on a model (the pipeline's quantization stage)."""

from __future__ import annotations

import copy

import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.calibration import calibrate_activations
from edge_ai_compression.compression.quantization.modules import fold_bn, quantize_model
from edge_ai_compression.compression.quantization.qat import QATConfig, quantization_aware_finetune
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.compression.quantization.reconstruction import (
    ReconstructionConfig,
    reconstruct,
)
from edge_ai_compression.core.experiment import QuantizationSection


def weight_spec(sec: QuantizationSection) -> WeightSpec:
    return WeightSpec(
        bits=sec.weight_bits,
        granularity=sec.granularity,
        group_size=sec.group_size,
        method=sec.weight_method,
    )


def run_quantization(
    model: nn.Module, sec: QuantizationSection, loader: DataLoader | None, device: str
) -> nn.Module:
    """Fold BN, swap in quantized layers, calibrate activations, then run the method."""
    fold_bn(model)
    fp_model = copy.deepcopy(model) if sec.method in ("adaround", "brecq") else None
    quantize_model(model, weight_spec(sec), sec.act_bits, first_last_bits=sec.first_last_bits)
    if sec.act_bits is not None:
        if loader is None:
            raise ValueError("activation quantization needs calibration data (a train loader)")
        cal = sec.calibration
        calibrate_activations(
            model,
            loader,
            method=str(cal["method"]),
            num_samples=int(cal["num_samples"]),
            percentile=float(cal["percentile"]),
            device=device,
        )
    if sec.method in ("adaround", "brecq"):
        if loader is None:
            raise ValueError(f"{sec.method} needs calibration data (a train loader)")
        assert fp_model is not None
        granularity = "layer" if sec.method == "adaround" else "block"
        reconstruct(
            model,
            fp_model,
            loader,
            ReconstructionConfig.from_dict(sec.options),
            granularity=granularity,
            device=device,
        )
    if sec.method == "qat":
        if loader is None:
            raise ValueError("qat needs training data (a train loader)")
        quantization_aware_finetune(model, loader, QATConfig.from_dict(sec.options), device)
    return model
