from __future__ import annotations

import copy
import csv
import json

import pytest
import torch
import torch.nn as nn

from edge_ai_compression.pretraining.config import TrainConfig
from edge_ai_compression.pretraining.signals import (
    SCALAR_FIELDS,
    SignalConfig,
    activation_signals,
    compute_signals,
    kurtosis,
    sharpness,
)
from edge_ai_compression.pretraining.trainer import run_training
from edge_ai_compression.utils.config_loader import load_yaml


def test_kurtosis_reference_values():
    g = torch.Generator().manual_seed(0)
    assert kurtosis(torch.rand(200_000, generator=g)) == pytest.approx(1.8, rel=0.02)
    assert kurtosis(torch.randn(200_000, generator=g)) == pytest.approx(3.0, rel=0.03)


def test_activation_channel_ratio_detects_outlier_channel():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 4))
    x = torch.randn(64, 16)
    base = activation_signals(m, x)["0"]["act_channel_ratio"]
    x[:, 3] *= 50
    assert activation_signals(m, x)["0"]["act_channel_ratio"] > 10 * base


def test_sharpness_positive_and_restores_weights():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    x, y = torch.randn(32, 8), torch.randint(0, 3, (32,))
    before = copy.deepcopy(m.state_dict())
    loss, sharp = sharpness(m, x, y, rho=0.05)
    assert sharp > 0 and loss > 0
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])


def test_compute_signals_all_fields_and_mode_restored():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 64, 4))
    m.train()
    x, y = torch.randn(16, 3, 8, 8), torch.randint(0, 4, (16,))
    scalars, per_layer = compute_signals(m, x, y, SignalConfig(hessian_iters=5))
    assert set(scalars) == set(SCALAR_FIELDS) and m.training
    assert set(per_layer) == {"0", "3"} and "hessian_trace" in per_layer["0"]
    assert all(v == v for k, v in scalars.items())  # no NaN when everything is enabled
    off, _ = compute_signals(m, x, y, SignalConfig(hessian=False, sharpness=False))
    assert off["hessian_trace"] != off["hessian_trace"] and off["probe_loss"] > 0
    with pytest.raises(ValueError, match="unknown signal options"):
        SignalConfig.from_dict({"sam": True})


def test_training_logs_signals_at_init_and_checkpoints(tmp_path):
    raw = load_yaml("edge_ai_compression/configs/train/smoke_train.yml")
    raw.update(
        checkpoint_dir=str(tmp_path / "ckpt"),
        results_dir=str(tmp_path / "r"),
        signals={"probe_samples": 8, "hessian_iters": 3},
        signal_every_steps=3,
    )
    result = run_training(TrainConfig.from_dict(raw))
    with open(tmp_path / "r" / "training_signals.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    assert steps[0] == 0 and 8 in steps and 3 in steps and 6 in steps
    assert all(r["run_id"] == result.run_id for r in rows)
    lines = (result.run_dir / "signals.jsonl").read_text().splitlines()
    assert len(lines) == len(rows) and "layers" in json.loads(lines[0])
