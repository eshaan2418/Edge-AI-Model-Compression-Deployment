# Deployment: export & hardware targets

How to turn a compressed model into a deployable artifact and reason about
whether it fits a target device.

## Exporting a model

```bash
python -m edge_ai_compression.export --model resnet18_cifar --format torchscript
python -m edge_ai_compression.export --model small_cnn_student --format json --out exports/summary.json
```

| Format | Extension | Status | Notes |
|--------|-----------|--------|-------|
| `torchscript` | `.pt` | ✅ native | `torch.jit.trace`; loadable with `torch.jit.load`. |
| `onnx` | `.onnx` | ⚠️ needs `onnx` | `torch.onnx.export`; raises an actionable error if `onnx`/`onnxscript` is missing. |
| `json` | `.json` | ✅ native | **Metadata only** — architecture, params, size, sparsity. Not weights, not runnable. |
| `tflite` | `.tflite` | ⛔ refused | PyTorch→TFLite needs an ONNX→TF→TFLite toolchain. The exporter refuses with guidance rather than faking a file. |

### Why TFLite is refused from PyTorch

TFLite is produced from TensorFlow/Keras graphs. There is no honest one-step
PyTorch→TFLite path in this repo. Two real options:

1. Use the **TF/Keras root scripts**, which export genuine `.tflite` models
   (dynamic-range and full-integer int8) from a Keras ResNet50 — see below.
2. Export to ONNX here, then convert ONNX→TF→TFLite with a tool like `onnx2tf`
   or `onnx-tf` (not bundled).

## Hardware profiles

`edge_ai_compression/hardware/profiles.py` defines resource **budgets** per
target:

| Profile | Max latency | Max size | Max RAM | Preferred export |
|---------|-------------|----------|---------|------------------|
| `cpu` | 100 ms | 500 MB | 4096 MB | torchscript |
| `raspberry_pi` | 200 ms | 100 MB | 512 MB | onnx |
| `smartphone` | 50 ms | 50 MB | 1024 MB | tflite |
| `microcontroller_sim` | 500 ms | 1 MB | 0.5 MB | tflite_micro |

> ⚠️ These are **documented planning budgets, not measurements** from physical
> devices. Confirm on the real target before shipping.

## Scoring feasibility

Score a benchmark/metrics JSON against a profile:

```bash
python -m edge_ai_compression.benchmarking.benchmark_model \
    --model resnet18_cifar --out results/benchmark.json

python -m edge_ai_compression.hardware.score \
    --metrics results/benchmark.json --profile raspberry_pi
```

Output reports:

- **feasible** — whether every constraint is satisfied.
- **utilization** — fraction of each budget used (latency/size/RAM).
- **violations** — which budgets were exceeded, by how much.
- **score** — accuracy reward minus resource + violation penalties (higher is
  better; infeasible always scores below feasible).

Metric names are flexible: `latency_ms` or `latency_ms_mean`, `ram_mb` or
`peak_ram_mib`, etc. Missing metrics are reported as `unknown` rather than
silently assumed.

## Real TFLite path (TensorFlow/Keras scripts)

```bash
pip install -e ".[tf-mot]"
python train_baseline.py --epochs 1 --limit 256 --weights none
python quantize_model.py --num-calibration-samples 50   # -> real .tflite artifacts
python benchmark.py --model-path models/quantized_integer_only.tflite --limit 200
```
