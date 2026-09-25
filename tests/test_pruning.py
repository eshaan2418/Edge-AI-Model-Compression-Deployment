from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import Bottleneck

from edge_ai_compression.compression.pruning.channel import prune_block_channels
from edge_ai_compression.compression.pruning.nm import apply_nm_pruning, nm_mask, satisfies_nm
from edge_ai_compression.compression.pruning.recovery import (
    LoRALayer,
    RecoveryConfig,
    finetune_masked,
    lora_recover,
    weight_sparsity,
)
from edge_ai_compression.core.experiment import CompressionConfig, PruningSection
from edge_ai_compression.core.pipeline import CompressionPipeline
from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.inference import kernels


def _resnet() -> nn.Module:
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():
        m(torch.randn(8, 3, 32, 32))
    return m.eval()


def test_nm_mask_exact_counts_and_skip():
    w = torch.randn(8, 16, 3, 3)
    mask = nm_mask(w, 2, 4)
    groups = mask.reshape(8, -1, 4)
    assert (groups.sum(dim=2) == 2).all()
    assert nm_mask(torch.randn(4, 3, 3, 3), 2, 4) is None  # K = 27
    with pytest.raises(ValueError):
        nm_mask(w, 4, 4)


def test_apply_nm_to_resnet_reports_skipped_stem():
    m = _resnet()
    report = apply_nm_pruning(m, 2, 4)
    assert report["conv1"].startswith("skipped") and report["layer1.0.conv1"] == "pruned"
    assert satisfies_nm(m.layer1[0].conv1.weight, 2, 4)
    assert 0.45 < weight_sparsity(m) < 0.5  # stem stays dense


@pytest.mark.skipif(not kernels.available() and not kernels.kernels_required(), reason="no kernels")
def test_nm_pattern_survives_bn_folding_into_sparse24_engine():
    from edge_ai_compression.inference.engine import compile_model

    m = _resnet()
    apply_nm_pruning(m, 2, 4)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        ref = m(x).numpy()
    out = compile_model(copy.deepcopy(m), "sparse24")(x.numpy())
    torch.testing.assert_close(torch.from_numpy(out), torch.from_numpy(ref), rtol=1e-3, atol=1e-3)


def test_channel_pruning_basicblock_shapes_and_params():
    m = _resnet()
    before = sum(p.numel() for p in m.parameters())
    n = prune_block_channels(m, 0.5, "l1")
    assert n == 8
    assert m.layer1[0].conv1.out_channels == 32 and m.layer1[0].conv2.in_channels == 32
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert sum(p.numel() for p in m.parameters()) < 0.6 * before


def test_channel_pruning_zero_amount_is_identity_and_bn_gamma_criterion():
    m = _resnet()
    ref = copy.deepcopy(m)
    prune_block_channels(m, 0.0)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(m(x), ref(x))
    g = _resnet()
    with torch.no_grad():
        g.layer1[0].bn1.weight.zero_()
        g.layer1[0].bn1.weight[5] = 10.0
    prune_block_channels(g, 0.98, "bn_gamma")  # keeps round(0.02 * 64) = 1 channel
    assert g.layer1[0].conv1.out_channels == 1


def test_channel_pruning_bottleneck_and_error_without_blocks():
    torch.manual_seed(0)
    block = nn.Sequential(Bottleneck(64, 16), Bottleneck(64, 16)).eval()
    x = torch.randn(1, 64, 8, 8)
    assert prune_block_channels(block, 0.5) == 2
    assert block[0].conv2.in_channels == 8 and block[0].conv3.in_channels == 8
    assert block(x).shape == (1, 64, 8, 8)
    with pytest.raises(ValueError, match="no BasicBlock"):
        prune_block_channels(nn.Sequential(nn.Linear(4, 4)), 0.5)


def _toy():
    torch.manual_seed(0)
    x = torch.randn(128, 16)
    y = (x[:, :2].sum(1) > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 2))
    return model, loader, x, y


def _acc(model, x, y):
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


@pytest.mark.parametrize("method", ["finetune", "lora"])
def test_recovery_keeps_mask_and_learns(method):
    model, loader, x, y = _toy()
    apply_nm_pruning(model, 2, 4)
    zeros = [(mod.weight == 0).clone() for mod in model if isinstance(mod, nn.Linear)]
    cfg = RecoveryConfig(method=method, epochs=40, lr=0.05, rank=4)
    fn = finetune_masked if method == "finetune" else lora_recover
    fn(model, loader, cfg, "cpu")
    linears = [mod for mod in model if isinstance(mod, nn.Linear)]
    assert all(isinstance(mod, nn.Linear) for mod in model if not isinstance(mod, nn.ReLU))
    for mod, z in zip(linears, zeros, strict=True):
        assert (mod.weight[z] == 0).all()  # pruned weights stay pruned
    assert _acc(model, x, y) > 0.9


def test_lora_adapter_is_small_and_masked():
    base = nn.Linear(128, 64)
    with torch.no_grad():
        base.weight[:, ::2] = 0
    lora = LoRALayer(base, rank=4, alpha=8)
    with torch.no_grad():
        lora.lora_b.normal_()
    assert (lora.delta()[:, ::2] == 0).all()
    assert lora.lora_a.numel() + lora.lora_b.numel() < 0.1 * base.weight.numel()


def test_pipeline_modes_and_config_validation():
    sec = PruningSection.from_dict({"enabled": True, "mode": "nm", "recovery": {"method": "lora"}})
    assert sec.tag == "nm:2:4+lora" and sec.target_sparsity == 0.5
    assert PruningSection.from_dict({"mode": "channel", "amount": 0.25}).tag == "channel:l1:0.25"
    with pytest.raises(ValueError, match="unknown pruning mode"):
        PruningSection.from_dict({"mode": "movement"})
    with pytest.raises(ValueError, match="unknown recovery keys"):
        PruningSection.from_dict({"recovery": {"epochz": 1}})
    m = _resnet()
    cfg = CompressionConfig(pruning=PruningSection(enabled=True, mode="channel", amount=0.5))
    out = CompressionPipeline(cfg, order=["prune"]).run(m, None, device="cpu")
    assert out.layer1[0].conv1.out_channels == 32
