# Pre-training variants and compressibility signals

Design rationale: [`DECISIONS.md`](DECISIONS.md) (Phase 5, D5.x). No results are quoted here.

## Trainer

`pretraining/trainer.py` (`train_model.py --config ... --set key=value`) does:

- SGD with Nesterov momentum or AdamW, linear warmup, then a cosine or constant schedule
- no weight decay on BN parameters, biases, or quantization scales
- `num_checkpoints` checkpoints at log-spaced steps, always including the last
- `export_path` to copy the final checkpoint somewhere stable

Every run is a row in `training_runs.csv`. Its config, history, fingerprint and signals are stored
under `results/training/<run_id>/`.

## Variants (`variant`, `variant_options`)

| Variant | Mechanism | Reference | Options (defaults) |
|---|---|---|---|
| `standard` | — | — | — |
| `quant_noise` | a random fraction p of each weight is fake-quantized per forward (straight-through); p = 1 is QAT from scratch; off in eval | Fan et al., ICLR 2021 | `bits: 4, p: 0.5` |
| `kurtosis` | loss += λ Σ_l (Kurt(W_l) − 1.8)² | Shkolnik et al., NeurIPS 2020 | `lam: 1.0, target: 1.8` |
| `rigl` | dynamic sparse training: ERK densities, drop smallest \|w\| / grow largest dense \|∇\|, cosine-decayed update fraction | Evci et al., ICML 2020 | `sparsity: 0.9, distribution: erk, delta_t: 100, alpha: 0.3, t_end: 0.75, dense_first_layer: true` |

## Signals (`signals: {...}`, `signal_every_steps`)

Signals are logged at initialization, at every checkpoint step, and every N steps, on a fixed
probe batch drawn once from the training data (so there is no test leakage). Each log adds a
row to `training_signals.csv`, with per-layer detail in the run's `signals.jsonl`.

| Signal | Meaning |
|---|---|
| `kurtosis_mean` / `_max` | weight tail heaviness (uniform 1.8, Gaussian 3) |
| `w_outlier_mean` / `_max` | max\|w\| / std(w), which sets the min-max quantization step |
| `weight_l2_mean` | weight norms |
| `act_outlier_mean` / `_max` | max\|x\| / std(x) at each layer input |
| `act_channel_ratio_mean` / `_max` | per-channel max / median at each layer input (outlier channels) |
| `hessian_trace` | Hutchinson trace of the loss Hessian (sum over layers) |
| `sharpness` | L(w + ρ·g/‖g‖) − L(w), SAM's first-order worst case, ρ = 0.05 |
| `probe_loss`, `weight_sparsity` | — |

These are the features for Phase 6's early-predictability study: given signals at step k,
predict the final model's post-compression accuracy.

## Model families

- ResNets: `resnet{10,18,34}_w{0.25,0.5,1.0}_cifar`, built from BasicBlocks with a CIFAR stem.
- ViTs: `vit_{t,s,m}_cifar`.

## Tracks

```bash
python -m edge_ai_compression.experiments.run_track --track pretrain_variants --smoke --seeds 0 \
    --results /tmp/r --models /tmp/m                                      # local smoke
python -m edge_ai_compression.experiments.run_track --track pretrain_variants --device cuda
```

| Track | Sweep | Runs |
|---|---|---|
| `pretrain_variants` | ResNet-18 × {standard, Quant-Noise p = 0.5, QAT p = 1, kurtosis, RigL 90%} | 15 |
| `pretrain_scaling` | 5 ResNet sizes × {standard, Quant-Noise} | 30 |
| `pretrain_length` | ResNet-18 w0.5 × {10, 30, 90} epochs (full schedule each) | 9 |
| `pretrain_vit` | ViT t / s / m | 9 |

## Results

- All Phase 5 training runs: [PENDING: run the four tracks in `notebooks/tracks.ipynb`]
