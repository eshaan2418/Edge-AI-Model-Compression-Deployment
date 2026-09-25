# Decision log

Non-trivial design and research decisions, newest last. Each entry: what was
decided, alternatives considered, and why.

---

## Phase 0: consolidation

### D0.1 Legacy code moves to `legacy/`, not deleted
- **Decision:** `src/model_compression/` → `legacy/model_compression/`, root TF scripts → `legacy/tflite/`.
- **Alternatives:** delete outright; keep in place.
- **Why:** the mission requires one pipeline, but the TFLite path is the only working int8 TFLite export today and a reference for Phase 2. Kept frozen, lint-checked, not installed.

### D0.2 Drop the `mc` console script; `legacy` is not packaged
- **Alternatives:** keep `mc` by adding `legacy*` to `packages.find`.
- **Why:** installing a top-level package called `legacy` (previously `src`) into site-packages pollutes the global namespace. Legacy code runs from the repo root with `python -m legacy.model_compression.cli`.

### D0.3 Placeholder benchmark row deleted
- `legacy/tflite/benchmark_results.md` contained an all-zero row that was never measured. Deleted and git-ignored (rule: no placeholder results).

### D0.4 Ruff pinned to 0.16.8 in both pre-commit and the `dev` extra
- **Why:** pre-commit used v0.8.4 while CI took the latest release, so formatting could diverge.

---

## Phase 1: benchmarking harness

### D1.1 Unit of replication is a fresh process, not an iteration
- **Decision:** CIs and hypothesis tests use per-process summaries (median latency per process). Per-iteration traces are only summarized descriptively (p50/p95/p99/max).
- **Alternatives:** bootstrap over iterations (what most benchmark scripts do); block bootstrap within one process.
- **Why:** iterations in one process are autocorrelated (caches, allocator, DVFS state), so treating them as i.i.d. gives CIs that are far too narrow. Between-process variation (address layout, scheduling, frequency) is often the dominant noise source. Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013); Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009).
- **Uncertainty:** percentile bootstrap under-covers with ~5 samples. Default `process_repeats: 5` gives indicative CIs; use ≥10 for headline claims.

### D1.2 Unknown and renamed `benchmark:` keys raise errors
- **Why:** renaming `latency_repeats` → `iters` would otherwise make old configs silently fall back to defaults.

### D1.3 GC disabled while timing
- Same as `timeit`. Collection pauses are Python runtime noise, not model cost.

### D1.4 Fingerprint split into hashed `static` and unhashed `dynamic` parts; git commit separate
- **Why:** runs on the same machine and software stack should group together (`fingerprint_hash`) even when power state or load differ between runs, or when the code commit changes. Power, thermal and load are still recorded for filtering.

### D1.5 Refuse to benchmark on battery or in Low Power Mode (`strict_environment: true` default)
- **Why:** macOS on battery changes scheduling and frequency policy. Smoke and test configs set `strict_environment: false`.

### D1.6 macOS has no CPU affinity; Linux does
- **Decision:** `cpu_affinity` is applied with `os.sched_setaffinity` on Linux (Colab/Kaggle/CI) and raises on macOS instead of being silently ignored.
- **Consequence:** M5 Pro numbers are "OS-scheduled across Super/Performance cores". Thread count is fixed explicitly (default 4 ≤ 6 Super cores) and recorded.

### D1.7 Benchmark input shape comes from the dataset, not the config
- **Decision:** config has `batch_size` only; spatial dims are taken from a real batch (the standalone CLI takes `--input-shape`).
- **Why:** a `(1,3,32,32)` default is silently wrong for Tiny ImageNet (64×64).

### D1.8 Memory metrics: whole-process peak, and peak above the torch runtime
- **Decision:** `peak_rss_mib` = process high-water mark (ru_maxrss). `model_peak_rss_mib` = peak minus the high-water mark right after `import torch`.
- **Alternatives:** "inference-only delta" (rss before vs after inference); a background RSS sampler.
- **Why:** ru_maxrss can't be reset on macOS, and checkpoint loading creates a transient buffer, so an inference-only delta is not measurable cleanly. A sampler misses short spikes. Both reported numbers are well-defined; the second is what the model adds (load transient + weights + activations).

### D1.9 Energy deferred; schema is ready for it
- `energy_j_per_inf` (nullable) and `energy_meter` columns exist now; only `NullMeter` is implemented. A powermetrics/RAPL sampler plugs in via `benchmarking/energy.py` without a schema change.

### D1.10 Benchmarking runs the model in a spawned `python -m ...worker` process
- **Alternatives:** `multiprocessing` spawn; in-process timing.
- **Why:** multiprocessing's spawn re-imports the parent's `__main__` module (and therefore torch) before the target runs, which contaminates cold-start stage timings. A plain subprocess gives a real cold start: spawn → interpreter → `import torch` → load → first inference.
- **Assumption:** `time.monotonic_ns()` is system-wide on macOS and Linux, so parent and child timestamps are comparable. Tested.

### D1.11 No separate benchmark CLI
- **Decision:** dropped the planned `benchmarking/cli.py`. `run_experiment.py` with all compression disabled (`smoke_benchmark.yml`) benchmarks a model and logs it to the DB.
- **Why:** a second entrypoint that benchmarks a model would be a parallel pipeline. Benchmarking exported artifacts (ONNX, C++ runtime) is designed in Phase 2 on top of the same worker.

### D1.12 `compression_order` records only the stages that ran
- **Why:** the column used to record the configured order (default `distill>prune>quantize`) even when only pruning was enabled, so DB rows mislabeled what was done. Unknown stage names now raise instead of being skipped.

### D1.13 Missing `checkpoint_in` is an error
- **Why:** the runner silently fell back to random weights when the checkpoint path didn't exist.

### D1.14 Quantized engine selected explicitly
- **Finding:** on torch 2.14 / macOS arm64 the quantized engine defaults to `none`, so the framework's dynamic quantization failed with `NoQEngine` on the primary platform. `utils/quant_engine.py` selects x86 > fbgemm > qnnpack.
- **Also noted:** torch 2.14 warns that quantized tensor dtypes (`qint8`, ...) are deprecated and will be removed. This shapes Phase 3: build quantization on simulated (fake-quant) float ops plus our own integer kernels (Phase 2), not on `torch.ao` quantized modules.

### D1.15 Linux peak memory comes from /proc VmHWM, not ru_maxrss
- **Finding (CI):** in a worker spawned from pytest on Linux, `ru_maxrss` read 1.12 GB before the worker allocated anything, while VmHWM read 27 MB. At `execve` the kernel folds the old (forked-from-parent) address space's high-water mark into the process's maxrss, so `ru_maxrss` reports the parent's footprint.
- **Decision:** Linux uses VmHWM (per address space, fresh at exec); macOS uses `ru_maxrss` (no /proc, no inheritance observed). Every Linux memory number would otherwise have been the parent process's size.

---

## Phase 2: C++ kernels + export

### Plan (self-approved under the autonomy rules)
- `csrc/`: standalone CMake project, one nanobind extension `edge_ai_compression.inference._C`. Single-threaded kernels with runtime ISA dispatch (scalar reference, NEON+dotprod, AVX2+FMA, AVX-512BW). Explicit ISA selection so tests can check every SIMD path against scalar.
  - `gemm_f32`: C[M,N] = A[M,K]·B[K,N] (+bias per row, optional ReLU). A (weights) pre-packed as 4-row panels `[M/4][K][4]`, 4×16 register tile, K-blocking.
  - `gemm_s8`: int8×int8→int32, bit-exact. NEON `vdotq_laneq_s32` on 4×4-interleaved panels; AVX2/AVX-512 sign-extend to int16 + `madd_epi16` (exact, no `maddubs` saturation).
  - `requantize`: int32 → f32 with per-row weight scale × activation scale + bias (+ReLU).
  - `gemm_w4`: int4 weight-only (symmetric, per-group scales), dequantized per K-block into the f32 panel format, then the f32 microkernel.
  - `gemm_sparse24`: 2:4 structured sparse weights (2 values + 2-bit indices per group of 4 along K).
  - `gemm_csr`: unstructured sparse weights (CSR), for the "why unstructured sparsity rarely speeds up" study.
  - `im2col` (f32, int8), `quantize_s8`, `absmax`, and microbenchmarks (peak FMA GFLOP/s, peak int8 GOP/s, triad bandwidth) for roofline.
- `edge_ai_compression/inference/`: Python wrappers, weight packing/quantization (numpy), an FX-based engine that runs ResNets on the kernels (BN folded), ONNX export + ONNX Runtime backend, a backend registry wired into the benchmark worker, roofline analysis, and a kernel-benchmark table in the experiment DB.
- Tests: bit-exact int8 vs numpy int32, tolerance for float; odd shapes, K=1, non-multiple-of-tile M/N/K, int8 extremes (overflow), non-contiguous inputs rejected; engine vs torch eager; ORT vs eager.

### D2.1 CMake project separate from the Python package build
- **Alternatives:** scikit-build-core as the package build backend (so `pip install -e .` compiles C++).
- **Why:** keeps `pip install -e .` fast and pure-Python for users who don't need kernels, and keeps the C++ build explicit and debuggable. CI runs `scripts/build_kernels.sh`, then the kernel tests with `EDGEAI_REQUIRE_KERNELS=1`, so a missing build fails CI instead of silently skipping.

### D2.2 Kernels are single-threaded (for now)
- **Why:** Apple clang ships without OpenMP, and a correct low-overhead thread pool is its own project. Kernel-vs-baseline comparisons therefore run everything at 1 thread (`torch.set_num_threads(1)`, ORT `intra_op_num_threads=1`), which is the fair comparison. Multi-threading is a later extension. This is a limitation for absolute latency claims.

### D2.3 Symmetric int8 with range [-127, 127]
- Weights per-output-channel, activations per-tensor (dynamic absmax in Phase 2; static calibration in Phase 3). Excluding -128 keeps quantization symmetric and makes negation safe. Costs 1 of 256 levels.

### D2.4 No TFLite / ExecuTorch backends
- **Why:** no edge device to run them on, and each is a large integration. ONNX Runtime is the strong, portable baseline. The legacy TFLite scripts remain for reference.

### D2.5 AVX-512 correctness via Intel SDE, gated on a repo variable
- **Why:** GitHub's x86 runners usually lack AVX-512, so the AVX-512 kernels would never execute in CI. Intel SDE emulates a Sapphire Rapids CPU. The action is pinned by commit SHA (it vendors the SDE binaries).
- **Gate:** using SDE means accepting Intel's license, which is the repo owner's decision, so the job runs only when `vars.ENABLE_SDE == 'true'` (see PROGRESS "NEEDS ESHAAN").
