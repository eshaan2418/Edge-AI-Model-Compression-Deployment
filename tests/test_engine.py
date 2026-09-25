from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.inference import kernels, packing

if not kernels.available():
    if kernels.kernels_required():
        raise RuntimeError("EDGEAI_REQUIRE_KERNELS=1 but kernels are not built")
    pytest.skip("C++ kernels not built", allow_module_level=True)

from edge_ai_compression.inference.engine import compile_model  # noqa: E402


def _resnet() -> nn.Module:
    torch.manual_seed(0)
    m = ModelRegistry.create("resnet18_cifar")
    m.train()
    with torch.no_grad():  # non-trivial BatchNorm statistics
        for _ in range(3):
            m(torch.randn(8, 3, 32, 32))
    return m.eval()


@pytest.fixture(scope="module")
def model():
    return _resnet()


@pytest.fixture(scope="module")
def x():
    return torch.randn(2, 3, 32, 32)


def _ref(m: nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return m(x).numpy()


def test_f32_matches_torch(model, x):
    out = compile_model(model, "f32")(x.numpy())
    np.testing.assert_allclose(out, _ref(model, x), rtol=1e-3, atol=1e-3)


def test_int8_close_to_torch(model, x):
    ref = _ref(model, x)
    out = compile_model(model, "int8")(x.numpy())
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < 0.05, rel


def test_w4_close_to_torch(model, x):
    ref = _ref(model, x)
    out = compile_model(model, "w4")(x.numpy())
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 0.3


def _prune_model(m: nn.Module, fn) -> nn.Module:
    import copy

    m = copy.deepcopy(m)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                w = mod.weight.reshape(mod.weight.shape[0], -1).numpy()
                mod.weight.copy_(torch.from_numpy(fn(w)).reshape(mod.weight.shape))
    return m


def test_sparse24_matches_2_4_pruned_torch_model(model, x):
    def to24(w: np.ndarray) -> np.ndarray:
        return w if w.shape[1] % 4 else packing.sparse24_to_dense(*packing.prune_24(w))

    pruned = _prune_model(model, to24)
    out = compile_model(model, "sparse24")(x.numpy())
    np.testing.assert_allclose(out, _ref(pruned, x), rtol=1e-3, atol=1e-3)


def test_csr_matches_torch_on_pruned_model(model, x):
    pruned = _prune_model(model, lambda w: packing.magnitude_prune(w, 0.6))
    out = compile_model(pruned, "csr")(x.numpy())
    np.testing.assert_allclose(out, _ref(pruned, x), rtol=1e-3, atol=1e-3)


def test_engine_pickles_without_packed_state(model, x):
    eng = compile_model(model, "int8").prepare()
    clone = pickle.loads(pickle.dumps(eng))
    assert all(n.gemm is None or n.gemm.packed is None for n in clone.nodes)
    np.testing.assert_array_equal(clone(x.numpy()), eng(x.numpy()))


def test_profile_records_every_op(model, x):
    out, records = compile_model(model, "f32").profile(x.numpy()[0])
    gemms = [r for r in records if r.kind == "gemm"]
    assert len(gemms) == 21  # 20 convs (BN folded) + 1 linear
    assert gemms[0].m == 64 and gemms[0].k == 27 and gemms[0].n == 32 * 32
    assert all(r.seconds > 0 for r in records)
    assert out.shape == (10,)


def test_bn_and_relu_are_fused(model):
    kinds = [n.kind for n in compile_model(model, "f32").nodes]
    assert "relu" not in kinds  # every ReLU fused into a conv or add
    assert kinds.count("add") == 8


def test_unsupported_op_raises():
    class Odd(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    with pytest.raises(NotImplementedError, match="unsupported"):
        compile_model(Odd())


def test_maxpool_and_standalone_relu():
    torch.manual_seed(1)
    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()
    x = torch.randn(1, 3, 8, 8)
    np.testing.assert_allclose(compile_model(m, "f32")(x.numpy()), _ref(m, x), rtol=1e-4, atol=1e-5)


def test_artifact_size_reflects_compressed_storage(model):
    sizes = {m: len(pickle.dumps(compile_model(model, m))) for m in ("f32", "int8", "w4")}
    assert 3.5 < sizes["f32"] / sizes["int8"] < 4.1
    assert sizes["f32"] / sizes["w4"] > 6
