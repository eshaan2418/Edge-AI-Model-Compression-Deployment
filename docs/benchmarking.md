# Benchmarking protocol

How every latency and memory number in the experiment DB is produced. Design
rationale is in [`DECISIONS.md`](DECISIONS.md) (D1.x).

## What one run measures

`Evaluator.evaluate()` does the following:

1. **Fingerprints the environment** (`benchmarking/fingerprint.py`): CPU model, core counts per
   performance level, ISA features, OS, Python, numpy/scipy/torch versions, torch BLAS and parallel
   backend, thread env vars, git commit + dirty flag, power source, Low Power Mode, thermal notes,
   load average. With `strict_environment: true` (the default) it refuses to run on battery or in
   Low Power Mode.
2. **Measures accuracy** on the configured device.
3. **Saves the model object** (`torch.save(model)`) and runs `process_repeats` **fresh Python
   processes** (`python -m edge_ai_compression.benchmarking.worker`). Each process:
   - applies `cpu_affinity` (Linux only; raises on macOS, which has no affinity API) and
     `torch.set_num_threads(num_threads)`
   - records cold-start stages: `startup` (spawn → interpreter), `import` (harness + `import torch`),
     `load` (`torch.load`), `first_inference`, and their sum `cold_start`
   - runs at least `warmup_iters` untimed iterations (and for at least `min_warmup_s`), then at
     least `iters` timed iterations (and for at least `min_time_s`), under `torch.inference_mode()`
     with the garbage collector disabled; each call is timed with `perf_counter_ns`
   - reports `peak_rss_mib` (process high-water mark) and `model_peak_rss_mib` (high-water mark
     minus the high-water mark right after `import torch`)
4. **Aggregates** (`benchmarking/report.py`):
   - `latency_median`: median of the per-process medians. **Use this for comparisons.**
   - `latency_median_ci_lo/hi`: percentile-bootstrap 95% CI over per-process medians (empty with
     one process).
   - `latency_mean/std/p50/p90/p95/p99/max`: all timed iterations pooled. Descriptive only;
     iterations within a process are autocorrelated, so these must not be used for CIs.
   - cold-start stages and memory: median across processes.

## Comparing two configurations

Use `benchmarking.stats.compare(a, b)` on the per-process medians of each run (stored in
`metrics.json → benchmark.process_medians_ms`). It reports the ratio of medians with a bootstrap
CI, a two-sided Mann-Whitney U p-value, and Cliff's delta. With 5 processes per side the smallest
achievable two-sided p is about 0.008. Percentile-bootstrap CIs under-cover at this sample size,
so use ≥10 processes for headline claims.

## Configuration (`benchmark:` section)

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | 1 | Input is `(batch_size, *sample_shape)`; sample shape comes from the dataset |
| `num_threads` | 4 | `torch.set_num_threads` in each worker (M5 Pro has 6 Super cores) |
| `warmup_iters` / `min_warmup_s` | 50 / 1.0 | Untimed warmup: both bounds must be met |
| `iters` / `min_time_s` | 1000 / 0.0 | Timed iterations: both bounds must be met |
| `process_repeats` | 5 | Independent processes (the unit of replication) |
| `cpu_affinity` | none | Core list, Linux only |
| `energy_meter` | `none` | Only `none` is implemented (energy columns stay empty) |
| `strict_environment` | true | Refuse to run on battery / Low Power Mode |

Unknown keys raise. Renamed keys (`latency_repeats`, `latency_warmup`, `input_shape`) raise with a
pointer to their replacement.

## Platform caveats

- **macOS (M5 Pro, primary platform):** no thread-to-core pinning and no frequency control. The OS
  schedules threads across Super and Performance cores. Numbers are reported as
  "Apple M5 Pro, OS-scheduled, N threads". Keep the machine on AC power, close other workloads,
  and check `load_avg` in `fingerprint.json`.
- **Linux cloud VMs (Colab/Kaggle, secondary):** shared and noisy. Pin cores with `cpu_affinity`,
  record the CPU model (the fingerprint does), and treat results as indicative.
- **Cold start** is a fresh process, but the OS file cache is not flushed (that needs root), so
  `load` after the first process reads a warm page cache.
- **Energy:** not measured yet. A powermetrics (macOS, needs sudo) or RAPL (x86 Linux) sampler
  plugs into `benchmarking/energy.py` without a schema change.
