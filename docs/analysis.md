# Analysis: studies and figures

Design rationale: [`DECISIONS.md`](DECISIONS.md) (Phase 6, D6.x). Every figure and table is produced by
`./reproduce.sh` from the experiment DB. `results/figures/MANIFEST.md` lists what exists and marks
each missing input as `[PENDING: run <config>]`.

## Data flow

```
Phase 5 training runs ──> training_runs.csv, training_signals.csv, checkpoints
          │
          └─ compress_runs (fixed panel) ──> experiments.csv rows with source_run_id/source_step
Phase 2/3/4 experiments + kernel study ──> experiments.csv, kernel_benchmarks.csv(.jsonl)
          │
          └─ ./reproduce.sh ──> results/figures/*.csv, *.png, MANIFEST.md
```

Compression panel (`experiments/compress_runs.py`): RTN W8A8, RTN W4A8, W4 group-32 weight-only,
2:4, 50% and 80% unstructured. All are training-free, so the accuracy drop measures the trained
model's *compressibility* and not the quality of a recovery procedure.

## Studies

| Study | Module | Question | Method |
|---|---|---|---|
| Early predictability | `analysis/early_prediction.py` | Can signals at training fraction f predict the final model's accuracy drop under compression? | RF / GBM / MLP / GP vs mean and params-only baselines; leave-one-configuration-out CV; cluster-bootstrap CIs over configurations (D6.1) |
| Signal ablations | same | Which signal group carries the information? | each group alone / all minus each group, GBM |
| Scaling | `analysis/scaling.py` | How does compressibility scale with parameters and epochs? | drop = a·(x/x₀)^−b + c; seed-bootstrap CI on b; AICc vs constant |
| Latency proxies | `analysis/latency_proxy.py` | Do FLOPs / params / sparsity predict latency? Can a learned proxy? | Spearman per backend and pooled; per-op kernel model summed over layers, calibrated leave-one-configuration-out |
| Pareto | `analysis/pareto_report.py` | Which configurations are non-dominated? | accuracy × latency × size (× energy when measured), per machine |
| Kernel study / roofline | `analysis/kernel_study.py`, `inference/roofline.py` | Where do kernels sit on the roofline; when does sparsity pay? | Phase 2 kernel benchmarks |

## Validation

Every analysis is tested on a synthetic DB with planted ground truth (`tests/synthetic_db.py`,
D6.2). The tests check that it recovers:

- a planted signal–target relationship, including its attribution to the right feature group
- a planted power-law exponent, and a flat series that must not beat the constant model
- planted per-kernel costs, where pooled FLOPs rank worse than the learned proxy

CI runs `./reproduce.sh` on the real-schema smoke DB and fails on analysis code errors.

## Results

[PENDING: run the Phase 5 tracks and `compress_runs` (notebooks/tracks.ipynb), the Phase 2 studies on
the M5 Pro, then `./reproduce.sh`.]
