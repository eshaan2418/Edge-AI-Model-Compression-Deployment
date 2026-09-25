"""Weight/activation quantization and sparse formats (numpy only, no compiled code).

These define the storage formats consumed by the C++ kernels and double as the
numpy reference implementations the kernels are tested against.
"""

from __future__ import annotations

import numpy as np

QMAX = 127  # symmetric int8: [-127, 127] (DECISIONS D2.3)


def quantize_rows_s8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8: ``w[m] ~= q[m] * scale[m]``. Zero rows get scale 1."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / QMAX, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -QMAX, QMAX).astype(np.int8)
    return q, scale


def quantize_tensor_s8(x: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 with absmax scale unless ``scale`` is given."""
    x = np.asarray(x, dtype=np.float32)
    if scale is None:
        amax = float(np.abs(x).max()) if x.size else 0.0
        scale = amax / QMAX if amax > 0 else 1.0
    q = np.clip(np.rint(x / np.float32(scale)), -QMAX, QMAX).astype(np.int8)
    return q, float(scale)


def ref_gemm_s8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact int32 reference for int8 GEMM."""
    return a.astype(np.int32) @ b.astype(np.int32)


# ------------------------------------------------------------------ int4 ----


def quantize_w4(w: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int4 per (row, group of ``group`` consecutive k).

    Returns ``q`` [M, ceil(K/2)] uint8 (element k in byte k//2, low nibble for even
    k, two's-complement nibbles in [-8, 7]; values are clipped to [-7, 7] so the
    grid is symmetric) and ``scales`` [M, ceil(K/group)] float32.
    """
    w = np.asarray(w, dtype=np.float32)
    m, k = w.shape
    n_groups = -(-k // group)
    padded = np.zeros((m, n_groups * group), dtype=np.float32)
    padded[:, :k] = w
    grouped = padded.reshape(m, n_groups, group)
    amax = np.abs(grouped).max(axis=2)
    scales = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(grouped / scales[:, :, None]), -7, 7).astype(np.int8)
    q = q.reshape(m, -1)[:, :k]
    return pack_nibbles(q), scales


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """int8 values in [-8, 7], shape [M, K] -> uint8 [M, ceil(K/2)]."""
    m, k = q.shape
    u = (q.astype(np.int16) & 0x0F).astype(np.uint8)
    if k % 2:
        u = np.concatenate([u, np.zeros((m, 1), dtype=np.uint8)], axis=1)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray, k: int) -> np.ndarray:
    lo = (packed & 0x0F).astype(np.int8)
    hi = (packed >> 4).astype(np.int8)
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    out[:, 0::2], out[:, 1::2] = lo, hi
    out = out[:, :k]
    return np.where(out >= 8, out - 16, out).astype(np.int8)


def dequantize_w4(q: np.ndarray, scales: np.ndarray, k: int, group: int) -> np.ndarray:
    vals = unpack_nibbles(q, k).astype(np.float32)
    return vals * np.repeat(scales, group, axis=1)[:, :k]


# ----------------------------------------------------------- 2:4 sparse ----


def prune_24(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the 2 largest-magnitude weights in every group of 4 along K.

    Returns ``values`` [M, K/2] float32 and ``meta`` [M, K/4] uint8
    (``idx0 | idx1 << 2`` with idx0 < idx1). K must be a multiple of 4.
    """
    w = np.asarray(w, dtype=np.float32)
    m, k = w.shape
    if k % 4:
        raise ValueError("2:4 sparsity needs K divisible by 4")
    groups = w.reshape(m, k // 4, 4)
    keep = np.sort(np.argsort(-np.abs(groups), axis=2, kind="stable")[:, :, :2], axis=2)
    values = np.take_along_axis(groups, keep, axis=2).reshape(m, k // 2)
    meta = (keep[:, :, 0] | (keep[:, :, 1] << 2)).astype(np.uint8)
    return values.astype(np.float32), meta


def sparse24_to_dense(values: np.ndarray, meta: np.ndarray) -> np.ndarray:
    m, half = values.shape
    k = half * 2
    dense = np.zeros((m, k // 4, 4), dtype=np.float32)
    idx = np.stack([meta & 3, (meta >> 2) & 3], axis=2).astype(np.int64)
    np.put_along_axis(dense, idx, values.reshape(m, k // 4, 2), axis=2)
    return dense.reshape(m, k)


# ------------------------------------------------------------------ CSR ----


def to_csr(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense -> CSR (values float32, col int32, row_ptr int32), dropping exact zeros."""
    w = np.asarray(w, dtype=np.float32)
    rows, cols = np.nonzero(w)
    row_ptr = np.zeros(w.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=w.shape[0]), out=row_ptr[1:])
    return w[rows, cols].astype(np.float32), cols.astype(np.int32), row_ptr


def magnitude_prune(w: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the ``sparsity`` fraction of smallest-magnitude weights (unstructured)."""
    w = np.asarray(w, dtype=np.float32)
    n_zero = int(round(sparsity * w.size))
    if n_zero == 0:
        return w.copy()
    order = np.argsort(np.abs(w), axis=None, kind="stable")
    out = w.copy().reshape(-1)
    out[order[:n_zero]] = 0.0
    return out.reshape(w.shape)
