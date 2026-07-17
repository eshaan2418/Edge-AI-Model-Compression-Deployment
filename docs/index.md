# Edge AI Model Compression & Deployment

Smaller, faster, still-accurate deep-learning models for edge devices — with a
reproducible research pipeline for searching the compression trade-off space.

Everything runs on a normal **CPU**, offline. No GPU, no downloads.

## Start here

```bash
pip install -e .
python -m edge_ai_compression.demo --quick
```

The demo prunes + quantizes a model and prints a size/latency trade-off table,
then writes `report.json` / `report.md` you can feed to the reporting tools below.

## Guides

| Guide | What it covers |
| ----- | -------------- |
| [Architecture](architecture.md) | How the framework is structured: registry, pipeline stages, experiment DB, Pareto search. |
| [Experiments](experiments.md) | Running compression experiments, sweeps, and search spaces. |
| [Analysis](analysis.md) | Statistical latency A/B testing, roofline analysis, layer sensitivity, and Pareto quality indicators — with their honesty caveats. |
| [Deployment](deployment.md) | Hardware-aware feasibility scoring, profile comparison, and honest export tooling. |
| [Recruiter demo](recruiter_demo.md) | A 60-second guided tour of what this project demonstrates. |

## Analysis & deployment CLIs

Every CLI supports `--help`. Common ones:

| Command | Purpose |
| ------- | ------- |
| `python -m edge_ai_compression.demo --quick` | Offline prune+quantize demo with a trade-off table. |
| `python -m edge_ai_compression.config.validate --config <cfg>` | Validate an experiment config (`--strict` rejects unknown keys). |
| `python -m edge_ai_compression.recipes.list` | List built-in compression recipes. |
| `python -m edge_ai_compression.recipes.apply --recipe <name> --config <base>` | Merge a recipe into a base config. |
| `python -m edge_ai_compression.analysis.search_space --config <sweep>` | Summarize a sweep's candidate count and dimensions. |
| `python -m edge_ai_compression.benchmarking.ab_compare --model-a <a> --model-b <b>` | Statistically compare two models' latency (bootstrap CIs, Mann–Whitney U, Cliff's delta, verdict). |
| `python -m edge_ai_compression.theory.roofline --model <m> --profile <p>` | Roofline / arithmetic-intensity analysis and theoretical latency floor. |
| `python -m edge_ai_compression.analysis.layer_sensitivity --model <m> --method prune` | Rank layers by output drift under per-layer pruning/quantization. |
| `python -m edge_ai_compression.optimization.indicators --results <csv>` | Score a Pareto front (hypervolume, epsilon-indicator, spacing). |
| `python -m edge_ai_compression.hardware.compare --metrics <report>` | Rank a model against all hardware profiles. |
| `python -m edge_ai_compression.reporting.generate_report --demo <report>` | Build a single-file HTML report. |
| `python -m edge_ai_compression.reporting.model_card --metrics <report>` | Generate an honest Markdown model card. |
| `python -m edge_ai_compression.repro.manifest --config <cfg>` | Capture a reproducibility manifest (git, env, config hash, seeds). |

## Honesty notes

- The offline demo uses **synthetic data**; any accuracy it prints is a
  plumbing/sanity number, **not** real model quality.
- Hardware profiles are **planning budgets**, not measurements from physical devices.
- Export tooling **refuses** rather than fabricates when a conversion isn't possible.

## Building these docs

The docs are plain Markdown and readable directly on GitHub. To serve them as a
site locally with [MkDocs](https://www.mkdocs.org/):

```bash
pip install -e ".[docs]"
mkdocs serve      # http://127.0.0.1:8000
```
