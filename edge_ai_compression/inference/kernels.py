"""Typed wrappers over the compiled kernels (``_C``).

Every wrapper converts inputs to C-contiguous arrays of the exact dtype and
allocates the output, so the strict (no implicit conversion) bindings never see
a bad array. ``isa`` is ``"auto"`` (best available) or an explicit ISA name.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import numpy as np

BUILD_HINT = 'pip install -e ".[kernels]" && scripts/build_kernels.sh'


def _load() -> ModuleType | None:
    try:
        from edge_ai_compression.inference import _C
    except ImportError:
        return None
    return _C


_C = _load()


def available() -> bool:
    return _C is not None


def require() -> ModuleType:
    if _C is None:
        raise ImportError(f"C++ kernels are not built. Run: {BUILD_HINT}")
    return _C


def kernels_required() -> bool:
    """CI sets EDGEAI_REQUIRE_KERNELS=1 so a missing build fails instead of skipping."""
    return os.environ.get("EDGEAI_REQUIRE_KERNELS") == "1"


def supported_isas() -> list[str]:
    return list(require().supported_isas())


def best_isa() -> str:
    return str(require().best_isa())


def _f32(x: Any) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _bias(bias: np.ndarray | None) -> np.ndarray | None:
    return None if bias is None else _f32(bias)


def _isa(isa: str) -> str:
    return best_isa() if isa == "auto" else isa


# ------------------------------------------------------------------ fp32 ----


def pack_f32(w: np.ndarray) -> Any:
    return require().pack_f32(_f32(w))


def gemm_f32(
    a: Any, b: np.ndarray, bias: np.ndarray | None = None, relu: bool = False, isa: str = "auto"
) -> np.ndarray:
    b = _f32(b)
    out = np.empty((a.M, b.shape[1]), dtype=np.float32)
    require().gemm_f32(_isa(isa), a, b, _bias(bias), relu, out)
    return out


# ------------------------------------------------------------------ int8 ----


def pack_s8(q: np.ndarray, isa: str = "auto") -> Any:
    return require().pack_s8(_isa(isa), np.ascontiguousarray(q, dtype=np.int8))


def gemm_s8(a: Any, b: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.int8)
    out = np.empty((a.M, b.shape[1]), dtype=np.int32)
    require().gemm_s8(a, b, out)
    return out


def requantize(
    acc: np.ndarray,
    row_scale: np.ndarray,
    col_scale: float,
    bias: np.ndarray | None = None,
    relu: bool = False,
) -> np.ndarray:
    acc = np.ascontiguousarray(acc, dtype=np.int32)
    out = np.empty(acc.shape, dtype=np.float32)
    require().requantize(acc, _f32(row_scale), float(col_scale), _bias(bias), relu, out)
    return out


def quantize_s8(x: np.ndarray, scale: float) -> np.ndarray:
    x = _f32(x)
    out = np.empty(x.shape, dtype=np.int8)
    require().quantize_s8(x, float(scale), out)
    return out


def absmax(x: np.ndarray) -> float:
    return float(require().absmax(_f32(x)))


# ------------------------------------------------------ int4 weight-only ----


def make_w4(q: np.ndarray, scales: np.ndarray, k: int, group: int) -> Any:
    """Build from ``packing.quantize_w4`` output."""
    return require().make_w4(
        np.ascontiguousarray(q, dtype=np.uint8), _f32(scales), int(k), int(group)
    )


def gemm_w4(
    a: Any, b: np.ndarray, bias: np.ndarray | None = None, relu: bool = False, isa: str = "auto"
) -> np.ndarray:
    b = _f32(b)
    out = np.empty((a.M, b.shape[1]), dtype=np.float32)
    require().gemm_w4(_isa(isa), a, b, _bias(bias), relu, out)
    return out


# ------------------------------------------------------------- sparse ----


def make_sparse24(values: np.ndarray, meta: np.ndarray) -> Any:
    return require().make_sparse24(_f32(values), np.ascontiguousarray(meta, dtype=np.uint8))


def gemm_sparse24(
    a: Any, b: np.ndarray, bias: np.ndarray | None = None, relu: bool = False, isa: str = "auto"
) -> np.ndarray:
    b = _f32(b)
    out = np.empty((a.M, b.shape[1]), dtype=np.float32)
    require().gemm_sparse24(_isa(isa), a, b, _bias(bias), relu, out)
    return out


def make_csr(values: np.ndarray, col: np.ndarray, row_ptr: np.ndarray, k: int) -> Any:
    """Build from ``packing.to_csr`` output."""
    return require().make_csr(
        _f32(values),
        np.ascontiguousarray(col, dtype=np.int32),
        np.ascontiguousarray(row_ptr, dtype=np.int32),
        int(k),
    )


def gemm_csr(
    a: Any, b: np.ndarray, bias: np.ndarray | None = None, relu: bool = False, isa: str = "auto"
) -> np.ndarray:
    b = _f32(b)
    out = np.empty((a.M, b.shape[1]), dtype=np.float32)
    require().gemm_csr(_isa(isa), a, b, _bias(bias), relu, out)
    return out


# ------------------------------------------------------------- im2col ----


def conv_out_size(size: int, k: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - k) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """[C, H, W] (float32 or int8) -> [C*kh*kw, Ho*Wo]."""
    dtype = np.int8 if np.asarray(x).dtype == np.int8 else np.float32
    x = np.ascontiguousarray(x, dtype=dtype)
    c, h, w = x.shape
    ho, wo = conv_out_size(h, kh, stride, pad), conv_out_size(w, kw, stride, pad)
    out = np.empty((c * kh * kw, ho * wo), dtype=dtype)
    fn = require().im2col_s8 if dtype == np.int8 else require().im2col_f32
    fn(x, kh, kw, stride, stride, pad, pad, out)
    return out


# ------------------------------------------------------ microbenchmarks ----


def peak_fma_gflops(isa: str = "auto", seconds: float = 0.5) -> float:
    return float(require().peak_fma_gflops(_isa(isa), seconds))


def peak_s8_gops(isa: str = "auto", seconds: float = 0.5) -> float:
    return float(require().peak_s8_gops(_isa(isa), seconds))


def triad_gbps(bytes_per_array: int = 256 * 1024 * 1024, reps: int = 5) -> float:
    return float(require().triad_gbps(bytes_per_array, reps))
