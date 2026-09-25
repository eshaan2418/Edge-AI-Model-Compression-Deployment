# Inference: C++ kernels, engine, backends, roofline

Design rationale is in [`DECISIONS.md`](DECISIONS.md) (D2.x). No results are quoted here. Every
number comes from the experiment DB (`kernel_benchmarks.csv`, `experiments.csv`).

## Build

```bash
pip install -e ".[kernels,export]"   # cmake, ninja, nanobind, onnx, onnxruntime, onnxscript
scripts/build_kernels.sh             # -> edge_ai_compression/inference/_C*.so
```

Tests that need the kernels skip when they aren't built, unless `EDGEAI_REQUIRE_KERNELS=1` (CI
sets it). CI builds and tests the kernels on Ubuntu x86-64 (AVX2 paths) and macOS arm64
(NEON paths). AVX-512 paths run under Intel SDE emulation once the repo variable `ENABLE_SDE` is
set (D2.5).

## Kernels (`csrc/`, single-threaded, runtime ISA dispatch)

All matrices are row-major. `C[M,N] = A[M,K] · B[K,N]`, where A holds weights (M = output
channels) and B holds activations (for convolution, B is the im2col matrix and N the number of
output pixels).

| Kernel | Storage | NEON | AVX2 / AVX-512 |
|---|---|---|---|
| `gemm_f32` | weights packed in 4-row panels | 4×16 tile, `vfmaq_laneq_f32` | 4×16 tile, FMA |
| `gemm_s8` (exact int32) | int8, ISA-specific panels | `vdotq_laneq_s32` on 4×4-interleaved panels | sign-extend + `madd_epi16` (no `maddubs` saturation) |
| `gemm_w4` | int4 nibbles + per-group fp32 scales | dequantize one K-block → fp32 microkernel | same |
| `gemm_sparse24` | 2 values + 2-bit indices per group of 4 | row-at-a-time axpy over 16 columns | same |
| `gemm_csr` | values + int32 column indices | row-at-a-time axpy over 16 columns | same |

The sparse kernels need one B row-segment load per FMA vector. The dense microkernel needs one
B load per four FMAs, because each A value is broadcast across a 4×16 tile. So halving the FLOPs
does not halve the time. Quantifying this is the job of the sparsity study.

Also included: `im2col` (fp32/int8, torch conv2d semantics), `quantize_s8` (round-half-even,
[-127, 127]), `requantize`, and microbenchmarks for roofline peaks (independent FMA / dot-product
chains; STREAM-style triad).

## Engine (`inference/engine.py`)

`compile_model(model, mode)` traces with torch.fx, folds BatchNorm into convs, fuses ReLU into
conv/linear/residual-add, and lowers convs to im2col + GEMM. Supported ops: Conv2d (groups=1,
dilation=1), BatchNorm2d after a conv, ReLU, residual add, MaxPool2d, global average pooling,
flatten, Linear, Identity/Dropout. Anything else raises `NotImplementedError`.

| Mode | Weights | Activations |
|---|---|---|
| `f32` | fp32 | fp32 |
| `int8` | per-output-channel symmetric int8 | per-tensor symmetric int8, dynamic (absmax per layer input), quantized before im2col |
| `w4` | int4 weight-only, group 32 | fp32 |
| `sparse24` | 2:4 (magnitude-pruned at compile time unless already 2:4; layers with K % 4 ≠ 0 stay fp32) | fp32 |
| `csr` | the model's existing zeros | fp32 |

Each layer stores only its compressed payload, so the pickled engine's size is the real storage
size.

## Backends (`inference/backends.py`)

`benchmark.backend` picks one of `torch_eager`, `onnxruntime`, `edge_f32`, `edge_int8`, `edge_w4`,
`edge_sparse24`, `edge_csr`. The evaluator:

1. exports one artifact for the chosen backend: `torch.save`, a single-file dynamo ONNX export, or
   a pickled engine
2. reports `size_mb` as the artifact's size on disk
3. computes accuracy with the same backend
4. benchmarks the artifact in fresh processes

ONNX Runtime runs with `intra_op_num_threads = num_threads`, sequential execution, and all graph
optimizations. Compare the C++ kernels against the others at `num_threads: 1` (D2.2).

## Kernel study and roofline

```bash
# Smoke (CI):
python run_kernel_study.py --config edge_ai_compression/configs/studies/smoke_kernel_study.yml
# Research run on the M5 Pro (AC power, idle machine):
python run_kernel_study.py --config edge_ai_compression/configs/studies/kernel_sparsity_m5.yml
python -m edge_ai_compression.analysis.kernel_study --results results --study kernel_sparsity_m5
```

Each measurement (fresh process per repeat, same protocol as model benchmarks) becomes a row in
`kernel_benchmarks.csv` with a run ID and fingerprint. `inference/roofline.py` takes machine peaks
from the logged `peak_*`/`triad` rows and computes, per GEMM row:

- arithmetic intensity from compulsory traffic, which is an upper bound (real traffic is higher)
- attainable throughput: `min(peak, AI × bandwidth)`
- efficiency
- whether the kernel is memory- or compute-bound

## Status of results

- Kernel study on the M5 Pro: [PENDING: run `configs/studies/kernel_sparsity_m5.yml` on AC power]
- Backend latency sweep: [PENDING: run `configs/sweeps/backend_latency_m5.yml` on AC power]
