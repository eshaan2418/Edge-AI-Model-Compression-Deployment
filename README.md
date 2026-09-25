# Is compressibility predictable early?

Does compression-awareness during training produce models that quantize and prune better
after training and run faster at inference? And can a model's final compressibility be
predicted early, from cheap training-time signals?

This repository is the full measured lifecycle for answering that:

```
train (variants, signals) → post-train (quantize / prune / recover) → lower to C++ kernels
      → benchmark (fresh processes, CIs, fingerprint) → analysis → paper
```

**Headline finding:** [PENDING: written only from `results/figures/` after the Phase 5/6 runs.
See `paper/` and `results/figures/MANIFEST.md`]

**Figure:** [PENDING: `results/figures/early_prediction_w4a8.png`, produced by `./reproduce.sh`]

Every number in this README, the paper and the blog traces to a run ID in the experiment DB.
Nothing is typed by hand.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kernels,export,viz]"
scripts/build_kernels.sh                 # C++ kernels (NEON / AVX2 / AVX-512)

./reproduce.sh                           # every figure + paper table from the experiment DB
```

The experiment DB (`results/`) is produced by:

| What | Command |
|---|---|
| Local smoke of any study track (seconds, synthetic data) | `python -m edge_ai_compression.experiments.run_track --track <track> --smoke --seeds 0 --results /tmp/r --models /tmp/m` |
| Training + PTQ/QAT ladder, pruning ladder, pre-training variants, scaling (GPU) | `notebooks/tracks.ipynb` on Colab/Kaggle, then `python -m edge_ai_compression.experiment_db.merge --src <downloaded>/results` |
| Compressibility targets for every training run | `python -m edge_ai_compression.experiments.compress_runs --device cuda` |
| Kernel study + backend latency (Apple M5 Pro, AC power) | `python run_kernel_study.py --config edge_ai_compression/configs/studies/kernel_sparsity_m5.yml` and `python launch_sweep.py --config edge_ai_compression/configs/sweeps/backend_latency_m5.yml` |
| One experiment | `python run_experiment.py --config <config.yml>` |

The runs still to be launched are listed under "NEEDS ESHAAN" in [`PROGRESS.md`](PROGRESS.md).

## Architecture

| Stage | Package | What it does | Guide |
|---|---|---|---|
| Training | `pretraining/` | SGD/AdamW trainer with log-spaced checkpoints; variants `standard`, `quant_noise`, `kurtosis`, `rigl`; compressibility signals (kurtosis, outlier ratios, Hessian trace, sharpness) logged to the DB | [pretraining](docs/pretraining.md) |
| Quantization | `compression/quantization/` | RTN + min-max/percentile/MSE calibration, AdaRound, BRECQ, LSQ QAT, HAWQ-style mixed precision, SmoothQuant; simulated layers that lower to the kernels exactly | [quantization](docs/quantization.md) |
| Pruning | `compression/pruning/` | unstructured, 2:4, channel pruning; masked fine-tuning or masked LoRA recovery | [pruning](docs/pruning.md) |
| Inference | `csrc/`, `inference/` | C++ kernels (fp32, exact int8, int4 weight-only, 2:4, CSR), FX engine, backends (PyTorch, ONNX Runtime, engine modes), roofline | [inference](docs/inference.md) |
| Benchmarking | `benchmarking/` | fresh-process timing, cold-start stages, peak memory, bootstrap CIs, Mann-Whitney comparisons, fingerprint | [benchmarking](docs/benchmarking.md) |
| Experiment DB | `experiment_db/` | the only results output: experiments, kernel benchmarks, training runs, training signals, per-run artifacts; merge from remote runs | |
| Analysis | `analysis/` | early predictability, signal ablations, scaling fits, latency proxies, Pareto frontiers, paper tables; each validated on synthetic data with planted answers | [analysis](docs/analysis.md) |
| Write-up | `paper/`, `docs/blog.md` | workshop paper and blog post, with PENDING markers wherever results are missing | |

Models: ResNets `resnet{10,18,34}_w{0.25,0.5,1.0}_cifar` (plus `resnet18_cifar`), ViTs
`vit_{t,s,m}_cifar`, torchvision ImageNet ResNet-18/50 for validation against the papers.
Datasets: CIFAR-10/100, Tiny ImageNet, ImageNet (license-gated), `fake` (offline smoke).

Other entrypoints (all take `--help`):
- `auto-compress`: constrained search over compression configs
- `search_compression.py`: Pareto frontier from the DB
- `train_surrogate.py`: RF/MLP/GP surrogates on DB rows
- `analyze_failures.py`: per-run confusion/failure cases
- `launch_sweep.py` / `resume_sweep.py`: resumable sweeps

Old code (the standalone PyTorch CLI and the TensorFlow/TFLite scripts) is frozen in
[`legacy/`](legacy/README.md).

## Engineering

- `ruff check .`, `ruff format --check .`, `pytest -q`. CPU only, no network, with a `fake`
  dataset smoke config for every module.
- CI runs on Ubuntu (AVX2 kernel paths) and macOS arm64 (NEON paths). It builds the kernels,
  runs every smoke config, runs `./reproduce.sh` on the smoke DB, and compiles the paper.
  Optionally it also runs the AVX-512 paths under Intel SDE (`ENABLE_SDE` repository variable).
- Design decisions and the reasoning behind them: [`docs/DECISIONS.md`](docs/DECISIONS.md).
  Interview-style Q&A per phase: [`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md).

## License

MIT.
