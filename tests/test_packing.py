from __future__ import annotations

import numpy as np
import pytest

from edge_ai_compression.inference import packing

RNG = np.random.default_rng(0)


def test_quantize_rows_s8_roundtrip_and_zero_rows():
    w = RNG.normal(size=(5, 33)).astype(np.float32)
    w[2] = 0.0
    q, s = packing.quantize_rows_s8(w)
    assert q.dtype == np.int8 and q.min() >= -127
    assert s[2] == 1.0 and not q[2].any()
    err = np.abs(q * s[:, None] - w).max(axis=1)
    assert (err <= s / 2 + 1e-7).all()


def test_quantize_tensor_s8_ties_round_half_even():
    q, s = packing.quantize_tensor_s8(np.array([0.5, 1.5, 2.5, -0.5], np.float32), scale=1.0)
    assert q.tolist() == [0, 2, 2, 0] and s == 1.0


@pytest.mark.parametrize("k,group", [(1, 32), (7, 4), (64, 32), (65, 32)])
def test_w4_pack_unpack_roundtrip(k, group):
    w = RNG.normal(size=(3, k)).astype(np.float32)
    q, scales = packing.quantize_w4(w, group)
    assert q.shape == (3, (k + 1) // 2) and scales.shape == (3, -(-k // group))
    vals = packing.unpack_nibbles(q, k)
    assert vals.min() >= -7 and vals.max() <= 7
    deq = packing.dequantize_w4(q, scales, k, group)
    step = np.repeat(scales, group, axis=1)[:, :k]
    assert (np.abs(deq - w) <= step / 2 + 1e-6).all()


def test_nibbles_cover_full_signed_range():
    q = np.array([[-8, -1, 0, 7, 3]], dtype=np.int8)
    assert np.array_equal(packing.unpack_nibbles(packing.pack_nibbles(q), 5), q)


def test_prune_24_keeps_two_largest_per_group():
    w = np.array([[1.0, -5.0, 2.0, 0.5, 0.0, 0.0, 3.0, -4.0]], np.float32)
    values, meta = packing.prune_24(w)
    assert values.tolist() == [[-5.0, 2.0, 3.0, -4.0]]
    assert meta.tolist() == [[1 | (2 << 2), 2 | (3 << 2)]]
    dense = packing.sparse24_to_dense(values, meta)
    assert np.array_equal(dense, np.array([[0, -5, 2, 0, 0, 0, 3, -4]], np.float32))
    with pytest.raises(ValueError):
        packing.prune_24(np.zeros((1, 6), np.float32))


def test_csr_roundtrip_and_empty_rows():
    w = packing.magnitude_prune(RNG.normal(size=(6, 10)).astype(np.float32), 0.7)
    w[3] = 0.0
    values, col, row_ptr = packing.to_csr(w)
    assert row_ptr[0] == 0 and row_ptr[-1] == len(values) == np.count_nonzero(w)
    dense = np.zeros_like(w)
    for r in range(6):
        dense[r, col[row_ptr[r] : row_ptr[r + 1]]] = values[row_ptr[r] : row_ptr[r + 1]]
    assert np.array_equal(dense, w)


def test_magnitude_prune_fraction():
    w = RNG.normal(size=(10, 10)).astype(np.float32)
    assert np.count_nonzero(packing.magnitude_prune(w, 0.3)) == 70
    assert np.array_equal(packing.magnitude_prune(w, 0.0), w)
