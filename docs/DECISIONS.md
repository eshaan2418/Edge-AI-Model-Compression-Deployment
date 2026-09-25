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

---

## Phase 3: PTQ / QAT ladder

### Plan (self-approved)
All methods quantize a **BN-folded** model (conv bias absorbs BN) into `QuantConv2d` / `QuantLinear` wrappers holding a weight quantizer (bits, granularity, scale, rounding) and an optional static activation quantizer. Accuracy is measured by simulation (fake-quant in float), and int8 / int4 models lower to the C++ engine for latency with the *same* integer weights and scales.

| Rung | Method | Reference |
|---|---|---|
| 1 | RTN weights, per-tensor and per-channel, 8/4 bits | baseline |
| 2 | Static W8A8 with activation calibration: min-max, percentile, MSE-optimal clipping | Krishnamoorthi 2018; Nagel et al. 2021 (white paper) |
| 3 | AdaRound: learned up/down rounding per weight, layer-wise output reconstruction | Nagel et al., ICML 2020 |
| 4 | BRECQ: block-wise (residual block) reconstruction with AdaRound rounding | Li et al., ICLR 2021 |
| 5 | QAT fine-tuning with LSQ learnable step sizes and STE | Esser et al., ICLR 2020 |
| 6 | HAWQ-style mixed precision: Hutchinson Hessian trace per layer → sensitivity × quant error → knapsack bit allocation {4, 8} under a size budget | Dong et al., HAWQ-V2 (NeurIPS 2020) |
| 7 | int4 weight-only, group-wise RTN (and AdaRound) | — |
| 8 | SmoothQuant for a small ViT: migrate activation outliers into weights before W8A8 | Xiao et al., ICML 2023 |

### D3.1 Build the supervised trainer now (`pretraining/`), not in Phase 5
- **Why:** the ladder needs trained baselines, and QAT needs a training loop; the framework had only a KD trainer. Phase 5 extends this trainer (compression-aware variants, signal logging) instead of adding a parallel one.

### D3.2 Simulated quantization + own kernels, not `torch.ao` quantized modules
- **Why:** torch 2.14 deprecates quantized tensor dtypes (D1.14). Fake-quant simulation is backend-independent and differentiable (needed by AdaRound/BRECQ/QAT). Deployment latency comes from the C++ engine, which consumes the same integer weights and scales.

### D3.3 Validation against papers: qualitative on CIFAR; ImageNet check needs data we don't have
- AdaRound / BRECQ / HAWQ report ImageNet numbers for torchvision ResNet-18/50. ImageNet validation requires a manual, license-gated download. We validate the *ordering* the papers report (at 4-bit weights: RTN ≪ AdaRound ≤ BRECQ; W8A8 ≈ FP) on CIFAR-10/100 and document the gap. An ImageNet validation config is provided for when the data is available (NEEDS ESHAAN).

### D3.4 No hyper-parameter tuning on the test set
- `build_loaders` has no validation split. Calibration uses training images; any tuned knob (percentile, λ, iterations) is set from the papers' defaults or on a held-out slice of the training set, never the test set.

### D3.5 Removed the torch.ao dynamic-quantization path
- `compression/quantization/dynamic.py`, `utils/quant_engine.py` and the `quantization.mode: dynamic_linear` key are gone (the key now raises). Quantization is always the simulated ladder (`method: rtn | ...`), and deployment goes through `edge_quant`.

### D3.6 Engine vs simulation: exact per layer, not bit-identical end to end
- **Finding:** C++ `quantize_s8` multiplied by `1/scale` while the simulation divides. Last-bit differences flipped rounding ties, and a W8A8 ResNet-18 differed by 1.6% relative at the logits. Fixed: the kernel now divides.
- **Remaining, intrinsic:** the engine accumulates exactly in int32 and rescales once; the simulation accumulates dequantized products in fp32. Those differ at ~1e-7, which occasionally flips a value sitting exactly on a rounding boundary at the next layer's input quantization. One-step differences then compound in a deep net (about 1% at the logits for a random-weight ResNet-18). The same happens between any two correct int8 implementations.
- **Test contract:** given identical inputs, every lowered layer matches its simulated layer to 1e-5 relative; end to end, relative error < 5% with identical argmax. Accuracy numbers are always measured with the backend that is benchmarked.

### D3.7 QAT keeps BatchNorm folded
- **Alternatives:** keep BN unfolded during QAT and fold at the end, freezing BN statistics after a few epochs (Krishnamoorthi 2018; Jacob et al. 2018).
- **Why:** starting from the calibrated, BN-folded PTQ model keeps one model representation across the ladder and makes the QAT result directly lowerable to the engine. The cost is that the model can't re-estimate BN statistics under quantization noise. Fine for short fine-tunes; a limitation for long QAT.

### D3.8 ViT track: three small ViTs; SmoothQuant only claimed if outliers exist
- `vit_t_cifar` / `vit_s_cifar` / `vit_m_cifar` (patch 4, dims 128/256/384, depths 6/6/8), explicit `qkv`/`proj`/`fc1`/`fc2` linears so every projection is quantizable. Attention matmuls (QKᵀ, AV) stay in float: SmoothQuant's W8A8 targets the linear layers.
- The C++ engine does not lower LayerNorm/GELU/attention, so ViT quantization is evaluated in simulation (accuracy). ViT latency comes from torch_eager/onnxruntime.
- `smoothquant.outlier_stats` (max/median of per-channel activation maxima at LayerNorm-fed linears) is logged before SmoothQuant results are interpreted. Systematic outliers are reported at LLM scale (Dettmers et al. 2022); at CIFAR-ViT scale SmoothQuant may be a no-op, which would be reported as a negative result.

---

## Phase 4: pruning + recovery

### Plan (self-approved)
`compression.pruning.mode`:
- `global_unstructured` (existing): global magnitude over all conv/linear weights.
- `layerwise_adaptive` (existing): per-layer sparsities from a scorer.
- `nm`: N:M semi-structured (default 2:4), keeping the N largest |w| in every group of M consecutive weights along the flattened input dim [in·kh·kw], the layout the engine uses. Layers whose K isn't divisible by M are left dense and reported. Mishra et al. 2021 ("Accelerating Sparse Deep Neural Networks", NVIDIA 2:4); Zhou et al. 2021 (N:M from scratch).
- `channel`: structured removal of the *inner* channels of each residual block (conv1 → conv2 in BasicBlock; conv1 → conv2 → conv3 in Bottleneck), which keeps residual shapes intact. Criterion is the L1 filter norm (Li et al., ICLR 2017) or |BN γ| (Liu et al., ICCV 2017, network slimming). The model is physically rebuilt smaller and dense, so every backend gets faster, not just sparse kernels.

`compression.pruning.recovery`:
- `finetune`: all weights trained, pruning masks enforced (masked SGD via `torch.nn.utils.prune` reparametrization).
- `lora`: frozen pruned weights plus low-rank adapters ΔW = B·A per conv/linear (Hu et al. 2021; B = 0 at init). The update is masked (W + M ⊙ BA) so merging keeps the sparsity pattern; an unmasked merge would densify the layer. Related: Zhang et al. 2023 (LoRAPrune) and Li et al. 2023 (LoSparse).
- Compared at equal steps; the parameter-efficiency trade-off is the result.

Engine lowering: `nm` 2:4 → `edge_sparse24` (pattern preserved through BN folding, since row scaling keeps within-row ranking); unstructured → `edge_csr`; channel → any dense backend.

### D4.1 `weight_sparsity` column (schema v4)
- The configured `pruning_sparsity` isn't what the model ends up with. It differs for channel pruning, skipped N:M layers, and first/last-layer policies. `weight_sparsity` is the measured fraction of zero conv/linear weights in the evaluated model, needed for the sparsity-vs-latency analysis.

### D4.2 Removed `compression/pruning/modes.py`
- An unused enum listing unimplemented modes (movement, lottery ticket). Valid modes are now validated by `PruningSection`.

---

## Phase 5: pre-training variants + compressibility signals

### Plan (self-approved)
**Trainer variants** (`train.variant`, options under `train.variant_options`):
- `standard`
- `quant_noise`: Quant-Noise (Fan et al., ICLR 2021). Each forward pass fake-quantizes a random fraction `p` of each conv/linear weight to `bits` bits (per channel, straight-through). `p = 1` is QAT from scratch.
- `kurtosis`: kurtosis regularization (Shkolnik et al., NeurIPS 2020). The loss gets λ Σ_l (Kurt(W_l) − K_T)², K_T = 1.8 (a uniform distribution), making weights uniform-like and robust to quantization.
- `rigl`: RigL dynamic sparse training (Evci et al., ICML 2020). Fixed overall sparsity with an ERK per-layer allocation. Every ΔT steps it drops the smallest-magnitude active weights and grows the same number of inactive weights with the largest dense-gradient magnitude, on a cosine-decayed update fraction that stops at 75% of training.

**Signals**, logged at every checkpoint step and every `signal_every_steps`, on a fixed probe batch:
- per-layer weight kurtosis
- weight outlier ratio (max|w| / std)
- weight norms
- activation outlier ratios (max|x| / std, and per-channel max / median as in SmoothQuant)
- Hutchinson Hessian trace
- SAM-style sharpness: L(w + ρ·g/‖g‖) − L(w), with ρ = 0.05 (Foret et al., ICLR 2021)
- probe loss

Rows go to `training_signals.csv` (run id, step, scalar summaries), with per-layer detail in the run's `signals.jsonl`. These are the surrogate features for Phase 6's early-predictability study: predict the *final* post-compression accuracy from signals at step k.

**Scaling family:** `resnet{depth}_w{width}_cifar` for depth ∈ {10, 18, 34} and width ∈ {0.25, 0.5, 1.0} (BasicBlock ResNets, CIFAR stem), plus the three ViTs. Training length is varied through `epochs`.

**Training sweeps:** a sweep with `kind: train` runs `run_training` per variant (same resume semantics).

### D5.1 RigL masks enforced by zeroing after each step, not by parametrization
- **Why:** growth needs the *dense* gradient ∂L/∂W, including inactive positions. With a forward on the dense weight tensor (inactive entries zero), autograd gives exactly that. Inactive weights and their momentum buffers are re-zeroed after every optimizer step. A mask parametrization would hide the inactive gradients.

### D5.2 Quant-Noise via parametrization
- Implemented with `torch.nn.utils.parametrize` (w + mask ⊙ (Q(w) − w).detach()), removed at the end of training with the original weights kept, so checkpoints are plain float models.

### D5.3 Experiment configs reject unknown top-level keys
- Every other section was already strict, but `ExperimentConfig` ignored unknown top-level keys. A typo such as `limit_sample` silently ran the full dataset, and a training sweep parsed as an experiment config passed the config tests. Now it raises.

### D5.4 Training sweeps (`kind: train`) with path templates
- A grid of seeds × variants needs distinct checkpoint paths. Top-level string values are formatted with the merged config (`models/pretrain/{model}_{variant}_e{epochs}_s{seed}.pt`), and the config test asserts no two runs in a training sweep share an export path.
