<div align="center">

# Hardware‑Aware Neural Network Compression & AutoML (Edge)

Smaller, faster, still‑accurate deep‑learning models for edge devices — with a
research pipeline for searching the compression trade‑off space.

</div>

The primary stack is a **PyTorch research framework** in `edge_ai_compression/`:
end‑to‑end compression experiments on CPU (prune / quantize / distill),
benchmarking, multi‑objective search, Pareto analysis, and experiment tracking.
Everything runs on a normal **CPU**; no GPU needed.

Earlier code (a standalone PyTorch ResNet‑18 CLI and the TensorFlow / Keras
TFLite scripts for ResNet50) is frozen under [`legacy/`](legacy/README.md).

---

## Quickstart (5 minutes, CPU‑only)

```bash
git clone https://github.com/eshaan2418/Edge-AI-Model-Compression-Deployment.git
cd Edge-AI-Model-Compression-Deployment

python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux  (Windows: .venv\Scripts\activate)
python -m pip install -U pip

# PyTorch research framework + dev tools (ruff, pytest):
pip install -e ".[dev]"

# Sanity check:
ruff check .
pytest -q                          # CPU only, no downloads

# Tiny end‑to‑end experiment — fully offline, no downloads, seconds on CPU:
python run_experiment.py --config edge_ai_compression/configs/experiments/smoke_cpu.yml
```

The smoke config (`smoke_cpu.yml`) uses a **synthetic `fake` dataset** (random
CIFAR‑shaped tensors — no files, no network), prunes a ResNet‑18 to 50%
sparsity, benchmarks it in fresh processes, and appends a row to the
experiment DB under `results/smoke/`. Swap `dataset: fake` → `dataset: cifar10`
in the config for a real run.

### Optional extras

```bash
pip install -e ".[kernels,export]" && scripts/build_kernels.sh   # C++ kernels + ONNX Runtime
pip install -e ".[tf-mot]"   # TensorFlow + tf-keras + tf-mot (legacy/tflite/ scripts)
pip install -e ".[viz]"      # matplotlib (plot_results.py)
pip install -e ".[all]"      # everything above
```

`requirements.txt` installs `-e .[dev]`; `environment.yml` creates a conda env
with `-e .[all]`. `pyproject.toml` is the single source of truth for
dependencies — the other two just point at it.

---

## PyTorch research framework (`edge_ai_compression`)

Run a compression experiment from a YAML config:

```bash
# Fast smoke tests (synthetic data, offline):
python run_experiment.py --config edge_ai_compression/configs/experiments/smoke_cpu.yml
python run_experiment.py --config edge_ai_compression/configs/experiments/smoke_benchmark.yml

# Full runs (download + use the whole dataset — slower):
python run_experiment.py --config edge_ai_compression/configs/experiments/baseline_eval.yml
python run_experiment.py --config edge_ai_compression/configs/experiments/prune_then_eval.yml
```

Two knobs keep runs cheap: `dataset: fake` (aliases `synthetic` / `debug` /
`random`) generates random CIFAR‑shaped tensors with no network access, and
`limit_samples: <N>` caps each split to its first N examples. Real datasets
(`cifar10`, `cifar100`, `tiny_imagenet`) are unchanged — drop both knobs for a
full experiment.

Other entrypoints (all support `--help`):

```bash
auto-compress --device cpu --max-latency-ms 50 \
  --max-size-mb 50 --min-accuracy 0.5 --budget 4     # constrained AutoML search (runs `budget` full experiments)
python search_compression.py --objective pareto \
  --results results/experiments.csv                   # Pareto frontier from logged runs
python train_surrogate.py --results results/experiments.csv --out results/surrogates
python analyze_failures.py --experiment-id <uuid>
python plot_results.py                                # needs .[viz]
python launch_sweep.py --config edge_ai_compression/configs/sweeps/full_compression_study.yml
python resume_sweep.py --sweep-id full_compression_study
python run_kernel_study.py --config edge_ai_compression/configs/studies/smoke_kernel_study.yml  # needs kernels
```

### What the framework implements

| Area | Capability |
|------|------------|
| **Experiment DB** | The only results output, always on: one row per run in `<results_dir>/experiments.csv` / `.jsonl` (schema v2) plus `artifacts/{id}/` (config, metrics, fingerprint, `model.pt`, per‑process latency traces, confusion matrix, failure cases). |
| **Benchmarking** | Latency and memory measured in fresh processes (default 5 × 1000 timed iterations after warmup); median of per‑process medians with bootstrap CI; cold‑start stages; peak RSS; hardware/software fingerprint per run. See [`docs/benchmarking.md`](docs/benchmarking.md). |
| **Inference** | C++ kernels (fp32, exact int8, int4 weight-only, 2:4 and CSR sparse; NEON / AVX2 / AVX‑512) with nanobind bindings; an FX engine that runs CNNs on them; backends `torch_eager`, `onnxruntime`, `edge_{f32,int8,w4,sparse24,csr}`; kernel study + roofline analysis. See [`docs/inference.md`](docs/inference.md). |
| **Pruning** | Global unstructured (`magnitude`) and `layerwise_adaptive` with `magnitude` / `gradient` / `activation` / `ablation` scorers. |
| **Quantization** | Dynamic linear (PyTorch). |
| **Distillation** | KD from a teacher checkpoint (temperature + alpha). |
| **Compression order** | `compression_order` / `compression_order_tag` (e.g. `distill>prune>quantize`). |
| **Multi‑objective** | Pareto frontier + NSGA‑II (`optimization/search/nsga2.py`). |
| **Surrogates / policy** | RF/MLP/GP surrogates on the results CSV; constraint‑feasible row selection. |
| **Diagnostics** | Confusion matrices, per‑class accuracy, ECE, NLL, failure cases. |
| **Datasets** | `cifar10`, `cifar100`, `tiny_imagenet`, and `fake` (synthetic, offline) via `build_loaders()`. |

Layout: `core`, `compression`, `optimization`, `benchmarking`, `inference`,
`analysis`, `data`, `utils`, `experiment_db`, `surrogate`, `policy`, `theory`,
`configs/`, `experiments/`, `auto_compress.py`.

---

## Legacy code

`legacy/` holds the old `src/model_compression/` PyTorch CLI (formerly `mc`)
and the TensorFlow / Keras TFLite scripts that used to live at the repo root.
It is not installed with the package and is kept for reference only; see
[`legacy/README.md`](legacy/README.md) for how to run it.

---

## Repository map

```
edge_ai_compression/     PyTorch research framework (models, data, compression, search, DB)
run_experiment.py search_compression.py train_surrogate.py ...   root wrappers
edge_ai_compression/configs/   experiment / dataset / search-space / sweep / study YAML
csrc/                    C++ kernels (CMake + nanobind), built by scripts/build_kernels.sh
tests/                   pytest suite (CPU, no network)
legacy/                  frozen PyTorch CLI + TF/TFLite scripts (not installed)
models/  data/  results/ generated artifacts (git-ignored)
```

Generated artifacts (`models/`, `data/`, dataset downloads, `results/artifacts`,
sweeps, plots, virtualenvs, caches) are **git‑ignored** — the repo stays lean.

---

## Development

```bash
ruff check .          # lint (also enforced in CI)
ruff format .         # auto-format
pytest -q             # tests
```

CI (`.github/workflows/ci.yml`) runs `ruff check .`, `ruff format --check .`,
`pytest -q`, and the offline smoke experiments on Python 3.11.

## License

MIT — see `LICENSE` if present.

## Acknowledgements

PyTorch & torchvision; TensorFlow / Keras, the TensorFlow Model Optimization
Toolkit (TF‑MOT), and TFLite for the deployment‑oriented quantization path.
