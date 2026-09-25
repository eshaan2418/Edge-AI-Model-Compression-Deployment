from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.signals import kurtosis
from edge_ai_compression.pretraining.trainer import clean_state_dict, run_training, train_model
from edge_ai_compression.pretraining.variants import (
    QuantNoise,
    RigL,
    erk_densities,
    make_variant,
)
from edge_ai_compression.utils.config_loader import load_yaml


def _toy():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
    )
    x = torch.randn(256, 32)
    y = (x[:, 0] > 0).long()
    return model, DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def test_make_variant_validation():
    assert isinstance(make_variant("rigl", {"sparsity": 0.8}), RigL)
    with pytest.raises(ValueError, match="unknown variant"):
        make_variant("lottery", {})
    with pytest.raises(ValueError, match="unknown rigl options"):
        make_variant("rigl", {"density": 0.1})
    with pytest.raises(ValueError, match="variant"):
        TrainConfig.from_dict({"variant": "movement"})


def test_quant_noise_active_only_in_training_and_removed_cleanly():
    model, _ = _toy()
    x = torch.randn(4, 32)
    qn = QuantNoise(bits=2, p=1.0)
    qn.setup(model, 10)
    model.train()
    noisy = model(x)
    model.eval()
    clean = model(x)
    assert not torch.allclose(noisy, clean)
    keys = clean_state_dict(model)
    assert "0.weight" in keys and not any("parametrizations" in k for k in keys)
    qn.finalize(model)
    torch.testing.assert_close(model(x), clean)
    assert isinstance(model[0].weight, nn.Parameter)


def test_kurtosis_regularizer_drives_weights_toward_uniform():
    model, loader = _toy()
    before = kurtosis(model[2].weight)
    cfg = TrainConfig(epochs=30, lr=0.05, variant="kurtosis", variant_options={"lam": 1.0})
    train_model(model, loader, cfg)
    after = kurtosis(model[2].weight)
    assert abs(after - 1.8) < abs(before - 1.8)


def test_erk_densities_meet_budget_and_cap():
    shapes = {"a": (16, 3, 3, 3), "b": (64, 16, 3, 3), "c": (10, 64)}
    dens = erk_densities(shapes, 0.9, dense={"a"})
    sizes = {n: torch.tensor(s).prod().item() for n, s in shapes.items()}
    overall = sum(dens[n] * sizes[n] for n in shapes) / sum(sizes.values())
    assert dens["a"] == 1.0 and all(0 < d <= 1 for d in dens.values())
    assert overall == pytest.approx(0.1, rel=1e-6)


def test_rigl_keeps_sparsity_and_moves_mask():
    model, loader = _toy()
    rigl = RigL(
        sparsity=0.8,
        distribution="uniform",
        delta_t=5,
        alpha=0.3,
        t_end=0.75,
        dense_first_layer=True,
        seed=0,
    )
    rigl.setup(model, total_steps=160)
    init = {k: v.clone() for k, v in rigl.masks.items()}
    assert init["0"].all()  # dense first layer
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    step = 0
    for _ in range(20):
        for x, y in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            rigl.after_backward(model, step + 1)
            opt.step()
            rigl.after_step(model, opt, step + 1)
            step += 1
    for name, mod in (("2", model[2]), ("4", model[4])):
        mask = rigl.masks[name]
        assert (mod.weight[mask == 0] == 0).all()  # inactive weights stay zero
        assert int(mask.sum()) == int(init[name].sum())  # drop count == grow count
        assert (opt.state[mod.weight]["momentum_buffer"][mask == 0] == 0).all()
    assert any(not torch.equal(rigl.masks[k], init[k]) for k in ("2", "4"))
    assert rigl.update_fraction(0) == pytest.approx(0.3)
    assert not rigl._is_update(int(0.75 * 160) + 5)  # no updates after t_end


@pytest.mark.parametrize(
    "variant,opts",
    [
        ("quant_noise", {"bits": 4, "p": 0.5}),
        ("kurtosis", {}),
        ("rigl", {"sparsity": 0.9, "delta_t": 2}),
    ],
)
def test_variant_training_runs_and_checkpoints_load_into_plain_model(tmp_path, variant, opts):
    from edge_ai_compression.core.registry import ModelRegistry

    raw = load_yaml("edge_ai_compression/configs/train/smoke_train.yml")
    raw.update(
        checkpoint_dir=str(tmp_path / "c"),
        results_dir=str(tmp_path / "r"),
        variant=variant,
        variant_options=opts,
    )
    result = run_training(TrainConfig.from_dict(raw))
    for path in result.checkpoints:
        state = torch.load(path, weights_only=False)["model_state"]
        ModelRegistry.create("resnet18_cifar").load_state_dict(state)
