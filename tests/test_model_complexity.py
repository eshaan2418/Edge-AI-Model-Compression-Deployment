from __future__ import annotations

import torch

from edge_ai_compression.compression.quantization.modules import quantize_model
from edge_ai_compression.compression.quantization.quantizer import WeightSpec
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.theory.model_complexity import estimate_flops_macs, layer_gemm_shapes


def test_resnet_shapes_and_flops_survive_quantization():
    m = ModelRegistry.create("resnet18_cifar").eval()
    shapes = layer_gemm_shapes(m, (1, 3, 32, 32))
    assert shapes[0] == {"name": "conv1", "kind": "conv", "m": 64, "n": 1024, "k": 27}
    assert shapes[-1] == {"name": "fc", "kind": "linear", "m": 10, "n": 1, "k": 512}
    flops = estimate_flops_macs(m, (2, 3, 32, 32))
    quantize_model(m, WeightSpec(bits=8), act_bits=None)
    assert estimate_flops_macs(m, (2, 3, 32, 32)) == flops > 0


def test_vit_token_linears():
    m = ModelRegistry.create("vit_t_cifar").eval()
    shapes = {s["name"]: s for s in layer_gemm_shapes(m, (1, 3, 32, 32))}
    assert shapes["blocks.0.attn.qkv"]["n"] == 65  # 64 patches + class token
    assert shapes["head"]["n"] == 1
    assert estimate_flops_macs(m, (1, 3, 32, 32)) > 0
    assert torch.is_grad_enabled()
