# legacy/

Frozen code from earlier iterations of this repo. It is kept for reference and
reproducibility, but **new work goes in `edge_ai_compression/`**. Nothing here
is installed by `pip install -e .`; run everything from the **repo root**.

| Path | What it is |
|------|------------|
| `model_compression/` | Small self-contained PyTorch ResNet-18 pipeline (train / prune / quantize / distill). Formerly `src/model_compression/` and the `mc` console script. |
| `scripts/` | Shell wrappers around `model_compression` (`run_all.sh` chains them). |
| `configs/` | Configs from the old trainer. Documentation only: nothing reads them. |
| `tflite/` | TensorFlow / Keras scripts that produce a ResNet50 CIFAR-10 baseline and TFLite artifacts (dynamic-range + full-integer int8). |

## PyTorch CLI (`model_compression`)

```bash
python -m legacy.model_compression.cli train --max-batches 5   # quick baseline train (smoke)
python -m legacy.model_compression.cli prune --help
python -m legacy.model_compression.cli quantize --help
python -m legacy.model_compression.cli distill --help
legacy/scripts/run_all.sh                                       # train -> prune -> quantize -> distill
```

These download CIFAR-10 into `data/` and write checkpoints to `models/`. The
import check in `tests/test_imports.py` is the only test coverage.

## TensorFlow / TFLite scripts (`tflite/`)

Install the TF extra first:

```bash
pip install -e ".[tf-mot]"
```

> **Keras 3 note:** these scripts target the Keras 2 SavedModel API and set
> `TF_USE_LEGACY_KERAS=1` at import time (satisfied by the `tf-keras` package
> that the `tf`/`tf-mot` extras install). No manual configuration needed.

Every script takes CLI flags and a `--limit` (or `--num-calibration-samples`)
smoke knob. Paths such as `models/...` are relative to the repo root:

```bash
# 1) Train a baseline -> models/baseline_model/  (--weights none skips ImageNet download)
python legacy/tflite/train_baseline.py --epochs 1 --limit 256 --weights none

# 2) Quantize -> models/quantized_dynamic_range.tflite + models/quantized_integer_only.tflite
python legacy/tflite/quantize_model.py --num-calibration-samples 50

# 3) Prune (TF-MOT) -> models/pruned_model/
python legacy/tflite/prune_model.py --epochs 1 --limit 256

# 4) Distill a compact student -> models/student_model/
python legacy/tflite/distill_model.py --epochs 1 --limit 256

# 5) Benchmark any SavedModel dir or .tflite file (size, latency, RAM, accuracy)
python legacy/tflite/benchmark.py --model-path models/baseline_model --limit 200
python legacy/tflite/benchmark.py --model-path models/quantized_integer_only.tflite --limit 200

# Combine pruning + full-integer quantization:
python legacy/tflite/combine_and_quantize.py --pruned models/pruned_model \
  --out models/combined_pruned_quantized.tflite --num-calibration-samples 50
```

`benchmark.py` appends a row to `legacy/tflite/benchmark_results.md` on each
run (git-ignored; created on first use). CI does not install TensorFlow, so
these scripts are lint-checked only.
