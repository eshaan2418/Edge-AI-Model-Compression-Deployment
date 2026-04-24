<div align="center">

# Hardware‑Aware Neural Network Compression & AutoML (Edge)

Smaller, faster, and accurate deep learning models for edge devices. The repository includes a **PyTorch research framework** (`edge_ai_compression`) with a compression pipeline, experiment runner, search (grid / random / Bayesian / evolutionary), Pareto analysis, benchmarking, and hardware-oriented entrypoints—alongside the original **TensorFlow/Keras** scripts for CIFAR‑10 baselines and TFLite workflows.

</div>

## Framework (`edge_ai_compression`)

Install the package (PyTorch stack is declared in `pyproject.toml`):

```bash
pip install -U pip
pip install -e .
```

Run an experiment from YAML, auto‑compression search, or hardware microbench:

```bash
python edge_ai_compression/experiments/run_experiment.py \
  --config edge_ai_compression/configs/experiments/prune_then_eval.yml

python edge_ai_compression/auto_compress.py \
  --device raspberry_pi \
  --max-latency-ms 50 \
  --max-size-mb 50 \
  --min-accuracy 0.5 \
  --budget 4

python edge_ai_compression/experiments/run_hardware_eval.py --model resnet18_cifar
```

Layout: `edge_ai_compression/core`, `compression`, `optimization`, `benchmarking`, `hardware`, `analysis`, `data`, `utils`, `experiment_db`, `surrogate`, `policy`, `theory`, `configs/`, `experiments/`, and `auto_compress.py`.

### Research system (what is implemented now)

| Area | Capability |
|------|----------------|
| **Experiment DB** | Every run can log to `results/experiments.csv`, `results/experiments.jsonl`, and `results/artifacts/{experiment_id}/` (`config.yaml`, `metrics.json`, `model.pt`, `latency_trace.npy`, `confusion_matrix.npy`, `failure_cases.json`, `model.tflite` placeholder). Enable with `experiment_db.enabled: true` in YAML (see `edge_ai_compression/configs/experiments/with_experiment_db.yml`). |
| **Layer-wise pruning** | `pruning.mode: layerwise_adaptive` with scorers `magnitude`, `gradient`, `activation`, `ablation` under `compression/pruning/layerwise/`. |
| **Compression order** | `compression_order` or `compression_order_tag` (e.g. `distill>prune>quantize`) drives stage order; see `edge_ai_compression/core/compression_orders.py`. |
| **Multi-objective** | Pareto frontier, NSGA-II (`optimization/search/nsga2.py`), `search_compression.py --objective pareto`. |
| **Surrogates / policy** | `train_surrogate.py` trains RF/MLP/GP on the CSV; `policy/supervised.py` picks a feasible row under constraints. |
| **Failure + calibration** | Confusion matrices, per-class accuracy, ECE, NLL, failure cases (`analysis/diagnostics.py`). |
| **Datasets** | `cifar10`, `cifar100`, `tiny_imagenet` via `build_loaders()`. |
| **Sweeps** | `launch_sweep.py` / `configs/sweeps/` (sequential runner; parallel workers are the next increment). |
| **Quality** | `.pre-commit-config.yaml` (ruff), `.github/workflows/ci.yml`. |

### Root CLI (Phase 16)

```bash
python train.py
python run_experiment.py --config edge_ai_compression/configs/experiments/prune_then_eval.yml
python launch_sweep.py --config configs/sweeps/full_compression_study.yml
python resume_sweep.py --sweep-id full_compression_study
python train_surrogate.py --results results/experiments.csv --out results/surrogates
python search_compression.py --objective pareto --results results/experiments.csv
python analyze_failures.py --experiment-id <uuid>
python plot_results.py   # needs: pip install -e '.[viz]'
```

## Highlights
- End‑to‑end baseline: ResNet50 with in‑model preprocessing and data augmentation
- Reproducible env: pinned `requirements.txt`, virtualenv, deterministic scripts
- Compression toolchain:
  - Pruning (TF‑MOT)
  - Post‑training quantization (dynamic/full‑integer, TFLite)
  - Knowledge distillation (custom `keras.Model` subclass)
- Unified benchmarking: size, latency, peak RAM, and accuracy, logged to Markdown

## Repository Structure
- `train_baseline.py` — Train and save the baseline ResNet50 model
- `benchmark.py` — Benchmark models (SavedModel or `.tflite`) and log results
- `prune_model.py` — Load baseline and prep for pruning (TF‑MOT)
- `quantize_model.py` — Create dynamic and full‑integer quantized TFLite models
- `distill_model.py` — Train a compact student via knowledge distillation and save it
- `benchmark_results.md` — Rolling log of benchmark runs as a Markdown table
- `models/` — Saved models and artifacts (ignored by git)
- `src/` `scripts/` `notebooks/` — Library code, automation, and analyses

## Setup
```bash
python3 -m venv venv            # or .venv
source venv/bin/activate        # macOS/Linux
pip install -U pip
pip install -r requirements.txt
```

## Baseline: Train ResNet50
```bash
source venv/bin/activate
python train_baseline.py
```
This will:
- Load CIFAR‑10
- Apply normalization + on‑the‑fly augmentation
- Build ResNet50 with a global‑avg‑pool + dense(10) head
- Compile with Adam + SparseCategoricalCrossentropy
- Print summary and train; then save to `models/baseline_model/` (SavedModel)

## Benchmarking
Benchmark any model (SavedModel dir or `.tflite`) for:
- Size (MB)
- Average single‑image latency (ms)
- Peak RAM (MiB)
- Accuracy on CIFAR‑10 test set

Configure the target in `benchmark.py` via `MODEL_PATH`, then run:
```bash
source venv/bin/activate
python benchmark.py
```
Results append to `benchmark_results.md`.

## Pruning (TF‑MOT)
```bash
source venv/bin/activate
python prune_model.py
```
Loads the baseline model and prepares it for TensorFlow Model Optimization Toolkit pruning workflows. Extend this script to wrap layers with `tfmot.sparsity.keras.prune_low_magnitude`, fine‑tune, then strip pruning wrappers before saving.

## Quantization (TFLite)
```bash
source venv/bin/activate
python quantize_model.py
```
Generates:
- Dynamic range quantized model: `models/quantized_dynamic_range.tflite`
- Full integer quantized model: `models/quantized_integer_only.tflite`

Benchmark both by pointing `MODEL_PATH` in `benchmark.py` to the respective file.

## Knowledge Distillation
```bash
source venv/bin/activate
python distill_model.py
```
Implements a custom `Distiller(keras.Model)` with overridden `train_step` to blend student loss and distillation loss (KLD over softened logits). Trains a compact CNN student and saves it to `models/student_model/` (and/or `models/distilled_student_model/`).

## Typical Workflow
1) Train baseline → `models/baseline_model/`
2) Quantize/prune/distill → produce artifacts under `models/`
3) Benchmark each → append to `benchmark_results.md`
4) Compare trade‑offs across size, speed, RAM, and accuracy

## Example Commands
```bash
# Baseline
python train_baseline.py && python benchmark.py

# Quantized (dynamic)
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/quantized_dynamic_range.tflite"|' benchmark.py
python benchmark.py

# Full‑integer TFLite
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/quantized_integer_only.tflite"|' benchmark.py
python benchmark.py

# Distilled student
sed -i '' 's|MODEL_PATH = ".*"|MODEL_PATH = "models/distilled_student_model"|' benchmark.py
python benchmark.py
```

## Reproducibility & Notes
- Environments: pinned in `requirements.txt`
- Artifacts: models under `models/` are ignored by git to keep the repo lean
- Benchmarks: durable record in `benchmark_results.md`

## License
MIT — see `LICENSE` if present. Otherwise, adapt as needed.

## Acknowledgements
- TensorFlow / Keras for training & SavedModel
- TensorFlow Model Optimization Toolkit (TF‑MOT) for pruning
- TFLite for deployment‑oriented quantization
