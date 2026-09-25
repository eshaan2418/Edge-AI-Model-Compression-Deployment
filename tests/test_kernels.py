"""Correctness of every compiled ISA path against numpy / torch references."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from edge_ai_compression.inference import kernels, packing

if not kernels.available():
    if kernels.kernels_required():
        raise RuntimeError(
            f"EDGEAI_REQUIRE_KERNELS=1 but kernels are not built ({kernels.BUILD_HINT})"
        )
    pytest.skip(f"C++ kernels not built ({kernels.BUILD_HINT})", allow_module_level=True)

C = kernels.require()
ISAS = kernels.supported_isas()
RNG = np.random.default_rng(0)
# (M, N, K): tile multiples, remainders in every dimension, degenerate sizes, long K.
SHAPES = [(1, 1, 1), (3, 5, 7), (4, 16, 4), (5, 17, 9), (17, 33, 257), (64, 100, 513), (8, 1, 300)]


def _rand_f32(*shape):
    return RNG.normal(size=shape).astype(np.float32)


def _rand_s8(*shape, lo=-127, hi=128):
    return RNG.integers(lo, hi, size=shape, dtype=np.int8)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("relu", [False, True])
def test_gemm_f32(isa, m, n, k, relu):
    a, b, bias = _rand_f32(m, k), _rand_f32(k, n), _rand_f32(m)
    ref = a @ b + bias[:, None]
    if relu:
        ref = np.maximum(ref, 0)
    out = kernels.gemm_f32(kernels.pack_f32(a), b, bias, relu, isa)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_gemm_s8_bit_exact(isa, m, n, k):
    a, b = _rand_s8(m, k, lo=-128), _rand_s8(k, n, lo=-128)
    out = kernels.gemm_s8(kernels.pack_s8(a, isa), b)
    assert out.dtype == np.int32
    assert np.array_equal(out, packing.ref_gemm_s8(a, b))


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("value", [-128, 127])
def test_gemm_s8_extremes_do_not_overflow_or_saturate(isa, value):
    # 4099 * 128 * 128 = 67,158,016 fits int32. An int16-saturating path
    # (e.g. maddubs) would fail here.
    k = 4099
    a = np.full((5, k), value, np.int8)
    b = np.full((k, 19), -128, np.int8)
    out = kernels.gemm_s8(kernels.pack_s8(a, isa), b)
    assert np.array_equal(out, packing.ref_gemm_s8(a, b))


def test_requantize_matches_numpy():
    acc = RNG.integers(-(2**20), 2**20, size=(7, 11), dtype=np.int32)
    rs, bias = RNG.uniform(1e-3, 1e-2, 7).astype(np.float32), _rand_f32(7)
    out = kernels.requantize(acc, rs, 0.05, bias, relu=True)
    ref = np.maximum(acc * (rs[:, None] * np.float32(0.05)) + bias[:, None], 0)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("group", [4, 32])
def test_gemm_w4(isa, m, n, k, group):
    w, b, bias = _rand_f32(m, k), _rand_f32(k, n), _rand_f32(m)
    q, scales = packing.quantize_w4(w, group)
    ref = packing.dequantize_w4(q, scales, k, group) @ b + bias[:, None]
    out = kernels.gemm_w4(kernels.make_w4(w, group), b, bias, False, isa)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("m,n,k", [(1, 1, 4), (3, 17, 8), (16, 33, 256), (7, 5, 516)])
@pytest.mark.parametrize("relu", [False, True])
def test_gemm_sparse24(isa, m, n, k, relu):
    values, meta = packing.prune_24(_rand_f32(m, k))
    b, bias = _rand_f32(k, n), _rand_f32(m)
    ref = packing.sparse24_to_dense(values, meta) @ b + bias[:, None]
    if relu:
        ref = np.maximum(ref, 0)
    out = kernels.gemm_sparse24(kernels.make_sparse24(values, meta), b, bias, relu, isa)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.95, 1.0])
def test_gemm_csr(isa, sparsity):
    w = packing.magnitude_prune(_rand_f32(13, 70), sparsity)
    b = _rand_f32(70, 37)
    out = kernels.gemm_csr(kernels.make_csr(w), b, None, False, isa)
    np.testing.assert_allclose(out, w @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "c,h,w,k,stride,pad",
    [
        (1, 5, 5, 3, 1, 1),
        (3, 32, 32, 3, 1, 1),
        (3, 32, 32, 7, 2, 3),
        (4, 9, 7, 1, 2, 0),
        (2, 8, 8, 3, 2, 1),
    ],
)
def test_im2col_matches_torch_unfold(c, h, w, k, stride, pad):
    x = _rand_f32(c, h, w)
    ref = F.unfold(torch.from_numpy(x)[None], k, padding=pad, stride=stride)[0].numpy()
    np.testing.assert_array_equal(kernels.im2col(x, k, k, stride, pad), ref)
    xq = _rand_s8(c, h, w)
    ref_q = kernels.im2col(xq.astype(np.float32), k, k, stride, pad)
    assert np.array_equal(kernels.im2col(xq, k, k, stride, pad), ref_q.astype(np.int8))


def test_quantize_s8_matches_numpy_including_ties():
    x = np.concatenate([_rand_f32(1000) * 3, np.array([0.5, 1.5, -2.5, 1e6, -1e6], np.float32)])
    for scale in (0.01, 1.0):
        ref, _ = packing.quantize_tensor_s8(x, scale)
        assert np.array_equal(kernels.quantize_s8(x, scale), ref)
    assert kernels.absmax(x) == np.abs(x).max()


def test_bindings_reject_noncontiguous_and_wrong_dtype():
    a = kernels.pack_f32(_rand_f32(4, 8))
    b = _rand_f32(8, 12)
    out = np.empty((4, 6), np.float32)
    with pytest.raises(TypeError):
        C.gemm_f32("scalar", a, b[:, ::2], None, False, out)  # non-contiguous input
    with pytest.raises(TypeError):
        C.gemm_f32("scalar", a, b.astype(np.float64), None, False, np.empty((4, 12), np.float32))
    with pytest.raises(TypeError):
        C.gemm_f32("scalar", a, b, None, False, np.empty((12, 4), np.float32).T)  # non-contig out


def test_wrappers_accept_noncontiguous_inputs():
    w = _rand_f32(6, 10)
    b = _rand_f32(10, 24)[:, ::2]
    np.testing.assert_allclose(
        kernels.gemm_f32(kernels.pack_f32(w), b), w @ b, rtol=1e-4, atol=1e-4
    )


def test_shape_and_isa_errors():
    a = kernels.pack_f32(_rand_f32(4, 8))
    with pytest.raises(ValueError, match="shape"):
        kernels.gemm_f32(a, _rand_f32(9, 3))
    with pytest.raises(ValueError, match="bias"):
        kernels.gemm_f32(a, _rand_f32(8, 3), bias=_rand_f32(5))
    unsupported = [i for i in ("neon", "avx2", "avx512") if i not in ISAS]
    for isa in unsupported:
        with pytest.raises(ValueError, match="not supported"):
            kernels.gemm_f32(a, _rand_f32(8, 3), isa=isa)
    with pytest.raises(ValueError, match="unknown isa"):
        kernels.gemm_f32(a, _rand_f32(8, 3), isa="sse9")
    with pytest.raises(ValueError, match="idx0 < idx1"):
        kernels.make_sparse24(_rand_f32(1, 2), np.array([[0]], np.uint8))


def test_microbenchmarks_positive():
    assert kernels.peak_fma_gflops(seconds=0.05) > 0
    assert kernels.peak_s8_gops(seconds=0.05) > 0
    assert kernels.triad_gbps(bytes_per_array=4 * 1024 * 1024, reps=2) > 0
