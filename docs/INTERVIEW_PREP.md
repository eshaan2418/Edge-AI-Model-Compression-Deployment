# Interview prep

Five hard questions per phase, with concise answers.

---

## Phase 0: consolidation

**1. Why move the old code to `legacy/` instead of deleting it? When would you delete it?**
The TF/TFLite scripts are the only working int8 TFLite export path and a reference for the Phase 2 export work. Keeping them frozen and lint-checked costs nothing. Delete once the unified PyTorch → ONNX export path reproduces what they do.

**2. Why drop the `mc` console script instead of packaging `legacy`?**
Packaging it would install a top-level module named `legacy` (previously `src`) into site-packages, where it can shadow or collide with other packages. Legacy code is reference material, not a library, so it runs from the repo root.

**3. The framework's quantization only converts `nn.Linear`. What does that mean for the existing ResNet-18 results?**
ResNet-18 has one Linear layer (the classifier). Nearly all compute is in convolutions, which stay FP32, so "quantized ResNet-18" results so far are close to a no-op. Static per-channel conv quantization is a prerequisite for any quantization claim (Phase 3).

**4. What does the CI smoke experiment catch that unit tests don't, and what can't it catch?**
It catches integration breaks: config parsing, data loading, the compression pipeline, evaluation and DB logging all wired together. It can't catch numerical correctness or performance regressions, because it uses random data and tiny iteration counts on a shared runner.

**5. How do you make sure every number in the README traces to a run?**
The experiment DB is the only results output. Every row has an experiment ID, and artifacts (config, metrics, fingerprint, traces) are stored under that ID. Figures are generated from the DB by a script. No hand-typed numbers; missing results are marked `[PENDING: run <config>]`.

---

## Phase 1: benchmarking harness

**1. Why is a process, not an iteration, your unit of replication? What goes wrong otherwise?**
Iterations in one process share cache state, allocator layout, JIT/dispatch warm state and CPU frequency, so they're autocorrelated. A bootstrap over 1000 iterations acts as if you had 1000 independent samples and gives a CI that's far too narrow. Much of the real variance is between processes (address-space layout, scheduling, which cores the OS picks). Kalibera & Jones (2013) formalize this as hierarchical repetition. I report pooled per-iteration percentiles only as descriptive distributions, and CIs and tests only over per-process medians.

**2. With 5 processes, how much should anyone trust your bootstrap CI?**
Only somewhat. The percentile bootstrap under-covers at n=5, and the smallest two-sided Mann-Whitney p with 5 vs 5 is about 0.008. So n=5 detects large effects but gives indicative intervals. For headline claims I'd raise `process_repeats` to 10 or more, or report an exact order-statistic interval for the median. The docs and DECISIONS D1.1 say this explicitly.

**3. You can't pin threads on macOS. How do you get defensible latency numbers on an M5 Pro?**
Control what I can and record what I can't. Fix the thread count below the Super-core count, refuse to run on battery or in Low Power Mode, disable GC during timing, run fresh processes, record load average and thermal notes in the fingerprint, and label numbers "OS-scheduled". For A-vs-B claims the remaining scheduler noise shows up as between-process variance, which the CI already includes. On Linux (Colab/CI) I can pin with `sched_setaffinity`.

**4. How do you measure peak memory, and why not "memory used by inference"?**
`ru_maxrss` gives the process high-water mark, but it can't be reset on macOS. Checkpoint loading also creates a transient buffer that can exceed the inference peak. So an "inference-only" delta isn't cleanly measurable, and a background RSS sampler misses short spikes. I report two well-defined numbers: the whole-process peak (what you'd need to deploy) and the peak above the torch runtime baseline (what this model adds, including load transients).

**5. What did building the harness uncover about the existing code?**
Four things:
- The old "peak RAM" was the whole pytest/Python process after importing torch (about 1.2 GB), not the model's footprint.
- "Energy" was a linear formula of latency and size, and it was a surrogate training target.
- On the primary platform, dynamic quantization didn't run at all: torch 2.14 on arm64 has no default quantized engine (`NoQEngine`).
- A missing checkpoint path silently benchmarked random weights.

All four are fixed and tested. The quantization finding also changes Phase 3's design, because torch has deprecated its quantized tensor dtypes.

---

## Phase 2: C++ kernels + export

**1. Your int8 GEMM on x86 doesn't use `_mm256_maddubs_epi16`, the standard fast path. Why not?**
`maddubs` multiplies unsigned by signed bytes and adds adjacent pairs into saturating int16. With symmetric int8 activations you'd have to shift them to unsigned (then correct with a row-sum term), and two products of about 128×127 already overflow int16. So it isn't exact. I sign-extend both operands to int16 and use `madd_epi16`, which sums pairs into int32 and is exact. That costs about 2× the multiply throughput of `maddubs`. VNNI (`vpdpbusd`) fixes the saturation but still needs unsigned activations. A test with K=4099 and all -128 operands would catch a saturating path.

**2. Walk me through the NEON int8 microkernel.**
`vdotq_laneq_s32(acc, b, a, lane)` adds, to each of 4 int32 lanes i, the dot product of 4 bytes of `b` (lane i) with 4 bytes of `a` selected by `lane`. I pack activations as 4 columns × 4 k-values per 16-byte vector and weights as 4 rows × 4 k-values per vector. With lane = row r, one instruction produces 4 output columns for row r. A 4×16 tile uses 16 accumulator registers, 4 B vectors and 1 A vector: 21 of 32 NEON registers, 16 dot instructions (256 multiply-adds) per 80 bytes loaded. Output rows are contiguous, so stores need no transpose.

**3. Why does 50% unstructured sparsity rarely beat dense on a CPU, and why can 2:4 do better?**
Dense GEMM reuses each loaded activation across a whole register tile: one B load feeds four FMAs, and weights are broadcast. A sparse kernel has to fetch the activation row that matches each nonzero, so it does one load per FMA vector plus index decoding, and it loses register blocking across rows. At 50% sparsity you save half the FLOPs but become load-bound, and CSR indices double the bytes per weight (a 50% CSR matrix is exactly as big as dense fp32; I measured this). 2:4 caps the metadata at 2 bits per value and makes every group the same shape, which is what the hardware sparse tensor cores exploit. On a CPU without such units the gain has to come from halved loads, so the crossover is an empirical question. My kernel study measures it per layer shape.

**4. How do you know your roofline analysis isn't flattering your kernels?**
Two choices make it conservative. Arithmetic intensity uses compulsory traffic (each operand moved once), which overstates AI, so "efficiency vs attainable" is measured against an optimistic roof. And the peaks come from single-core microbenchmarks on the same machine (independent FMA / dot chains, STREAM-style triad), logged with the same fingerprint as the kernel rows, so I never compare against spec-sheet numbers. The analysis refuses to mix machines.

**5. How does a benchmark of your C++ engine stay comparable with PyTorch and ONNX Runtime?**
Every backend goes through the same path:
- Export one artifact, then time it in fresh processes with the same warmup/iteration protocol.
- The same input shape, 1 thread for all (my kernels are single-threaded; ORT gets `intra_op_num_threads=1`).
- Accuracy measured with the backend itself.
- `size_mb` = the artifact on disk. The engine stores compressed payloads, so int8/int4 size is real.
- Cold start includes loading and packing weights, as it would for a real runtime.

Two bugs this caught before any results were produced: the dynamo ONNX exporter writing weights to a side file (so size looked like 0.1 MB), and Linux `ru_maxrss` inheriting the parent's peak across exec.
