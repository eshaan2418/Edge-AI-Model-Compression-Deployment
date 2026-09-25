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
