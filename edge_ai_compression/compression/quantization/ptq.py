"""Run one rung of the PTQ/QAT ladder on a model (the pipeline's quantization stage)."""

from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_compression.compression.quantization.calibration import calibrate_activations
from edge_ai_compression.compression.quantization.modules import fold_bn, quantize_model
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
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
    """Fold BN, swap in quantized layers, and calibrate activations (in place)."""
    fold_bn(model)
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
    return model
