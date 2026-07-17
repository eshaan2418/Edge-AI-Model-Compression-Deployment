# A 5-minute guided tour

For a reviewer who wants to see the project work end-to-end, on a laptop, with no
GPU and no dataset downloads. Every command below is CPU-only and offline.

## 0. Setup (~1 min)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 1. The offline demo (~30s)

```bash
python -m edge_ai_compression.demo --quick
```

You'll get a trade-off table comparing a baseline model to a pruned +
dynamically-quantized version — parameters, size, latency, and (clearly labeled)
synthetic accuracy — plus JSON/Markdown reports in `results/demo/`.

> The `synthetic_accuracy` column is measured on **randomly labeled fake data**.
> It proves the pipeline runs; it is *not* a quality metric. This labeling is
> deliberate — the project never dresses up synthetic numbers as real results.

## 2. Benchmark a real architecture (~20s)

```bash
python -m edge_ai_compression.benchmarking.benchmark_model \
    --model resnet18_cifar --repeats 20 --out results/benchmark.json
```

Latency percentiles (p50/p95/p99), throughput, parameter/size/FLOP estimates,
and peak RAM — all measured on synthetic input, so it's the model's *compute*
profile with no dataset involved.

## 3. Is it deployable? (~5s)

```bash
python -m edge_ai_compression.hardware.score \
    --metrics results/benchmark.json --profile raspberry_pi
python -m edge_ai_compression.hardware.score \
    --metrics results/benchmark.json --profile microcontroller_sim
```

The Raspberry Pi profile likely passes; the microcontroller profile likely
reports violations. That contrast *is* the point: the "best" model depends on the
target device's budget.

## 4. Export it (~5s)

```bash
python -m edge_ai_compression.export --model resnet18_cifar --format torchscript
python -m edge_ai_compression.export --model resnet18_cifar --format tflite
```

The TorchScript export succeeds; the TFLite export **refuses with guidance**
(PyTorch→TFLite needs a TF toolchain). Honest tooling over silent placeholders.

## 5. Run a tracked experiment (~5s)

```bash
python run_experiment.py \
    --config edge_ai_compression/configs/experiments/smoke_cpu.yml
```

A full config-driven run: build model → prune → evaluate → log. Swap
`dataset: fake` → `dataset: cifar10` for the real thing.

## What to look at in the code

- `edge_ai_compression/core/pipeline.py` — the `CompressionStage` abstraction.
- `edge_ai_compression/optimization/pareto/` — multi-objective frontier logic.
- `edge_ai_compression/hardware/scoring.py` — feasibility scoring.
- `edge_ai_compression/interfaces.py` — the typed component contracts.
- `tests/` — CPU-only, offline, fast.

See [architecture.md](architecture.md) for the full component map.
