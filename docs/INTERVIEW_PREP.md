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
