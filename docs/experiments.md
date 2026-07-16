# Experiments & the experiment database

How to run compression experiments and track them reproducibly.

## Running an experiment

Experiments are driven by YAML configs under
`edge_ai_compression/configs/experiments/`:

```bash
# Offline smoke run (synthetic data, ~seconds):
python run_experiment.py --config edge_ai_compression/configs/experiments/smoke_cpu.yml

# Real data (downloads CIFAR on first run):
python run_experiment.py --config edge_ai_compression/configs/experiments/baseline_eval.yml
python run_experiment.py --config edge_ai_compression/configs/experiments/prune_then_eval.yml
```

### Two knobs that keep runs cheap

- `dataset: fake` (aliases `synthetic` / `debug` / `random`) — random
  CIFAR-shaped tensors, no download, no network. Used by CI and the demo.
- `limit_samples: <N>` — cap each split to its first N examples for a fast
  partial run on real data.

Drop both for a full experiment on `cifar10` / `cifar100` / `tiny_imagenet`.

## The experiment database

Enable logging with `experiment_db.enabled: true` (see
`configs/experiments/with_experiment_db.yml`). Each run appends to:

```
results/
  experiments.csv      # one row per run, fixed schema (see experiment_db/record.py)
  experiments.jsonl    # same runs, full nested detail
  artifacts/<id>/
    config.(json|yml)
    metrics.json
    environment.json     # python, platform, package versions, git commit, device
    model_summary.json   # params, size_mb, sparsity, layer histogram
    model.pt             # (optional) the model
    latency_trace.json, confusion_matrix.*, failure_cases.*
```

`environment.json` and `model_summary.json` make a run reproducible and
self-describing: you can tell *what ran, on what, with which library versions*
months later.

## Analyzing results

Once you have runs logged in `results/experiments.csv`:

```bash
# Non-dominated configurations across accuracy vs. latency/size/RAM:
python -m edge_ai_compression.analysis.pareto \
    --results results/experiments.csv --out results/pareto --plot

# Multi-objective search / Pareto reporting (legacy entry point, still supported):
python search_compression.py --objective pareto --results results/experiments.csv

# Constrained AutoML search (runs `budget` full experiments):
python auto_compress.py --device cpu --max-latency-ms 50 --max-size-mb 50 \
    --min-accuracy 0.5 --budget 4

# Surrogate models over the results table:
python train_surrogate.py --results results/experiments.csv --out results/surrogates

# Failure analysis for one run:
python analyze_failures.py --experiment-id <uuid>
```

The Pareto CLI only uses objective columns that are actually present and
non-null, so partially populated result tables work without crashing.

## Reproducibility checklist

- [ ] Config committed (or captured in `artifacts/<id>/config.*`).
- [ ] `experiment_db.enabled: true` so `environment.json` is captured.
- [ ] Real vs. synthetic data noted — never report `fake`-dataset accuracy as a
      quality result.
- [ ] Results (`results/`, `models/`, datasets) stay git-ignored.
