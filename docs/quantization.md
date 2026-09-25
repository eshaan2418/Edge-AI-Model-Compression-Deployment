# Quantization: the PTQ/QAT ladder

Design rationale: [`DECISIONS.md`](DECISIONS.md) (Phase 3, D3.x). No results are quoted here.
Everything comes from the experiment DB, and missing results are marked PENDING.

## How a rung runs

`compression.quantization` in an experiment config selects the rung. `run_quantization`:

1. folds BatchNorm into the preceding convolutions (`fold_bn`)
2. applies SmoothQuant smoothing (`method: smoothquant` only)
3. computes mixed-precision bits from Hessian traces (`method: hawq` only)
4. swaps every Conv2d/Linear for `QuantConv2d`/`QuantLinear`: symmetric narrow-range grids,
   weight scales per tensor, per channel, or per group; first/last layers at `first_last_bits`
5. calibrates static per-tensor input scales if `act_bits` is set (min-max, percentile, or MSE)
6. learns rounding (AdaRound: per layer; BRECQ: per residual block) or fine-tunes with LSQ
   (`qat`)

The result is a simulated (fake-quant) model. With backend `edge_quant` the C++ engine runs the
same integer weights and scales. Every lowered layer matches its simulation to 1e-5
relative given identical inputs (tested). End to end the two agree up to rounding-boundary
flips (D3.6).

| `method` | What it does | Reference | Options key |
|---|---|---|---|
| `rtn` | round-to-nearest weights (+ calibrated activations) | baseline | — |
| `adaround` | learned rounding, layer-wise reconstruction | Nagel et al., ICML 2020 | `adaround: {iters, lr, reg_weight, batch_size, num_samples, ...}` |
| `brecq` | learned rounding, block-wise reconstruction, optional Fisher weighting | Li et al., ICLR 2021 | `brecq: {..., fisher}` |
| `qat` | LSQ learnable step sizes + STE fine-tuning | Esser et al., ICLR 2020 | `qat: {epochs, lr, momentum, weight_decay, max_steps}` |
| `hawq` | Hutchinson Hessian traces → ILP bit allocation {4, 8} under an average-bit budget | Dong et al., NeurIPS 2020 | `hawq: {avg_bits, candidate_bits, hutchinson_iters, hutchinson_samples, tol}` |
| `smoothquant` | migrate activation outliers into weights (ViT) | Xiao et al., ICML 2023 | `smoothquant: {alpha, num_samples}` |

Other keys: `weight_bits`, `act_bits` (null = weight-only), `granularity`
(`per_tensor` / `per_channel` / `per_group` + `group_size`), `weight_method` (`minmax` / `mse`),
`first_last_bits`, `calibration: {method, num_samples, percentile}`.

Engine support (`edge_quant`): W≤8 A8 per-tensor/channel → int8 kernel (4-bit codes run exactly,
without the storage saving); W4 weight-only → int4 kernel; W8 weight-only → fp32 on dequantized
weights. Anything else raises. ViTs are simulation-only (D3.8).

## Running the ladder

```bash
# Local smoke of the whole Phase 3 flow (synthetic data):
python -m edge_ai_compression.experiments.run_phase3 --smoke --seeds 0 --results /tmp/r --models /tmp/m
# Real runs (Colab/Kaggle GPU): notebooks/phase3_ladder.ipynb, or directly:
python -m edge_ai_compression.experiments.run_phase3 --track resnet18 --device cuda
python -m edge_ai_compression.experiments.run_phase3 --track vit_s --device cuda
```

Sweeps: `configs/sweeps/ptq_ladder_resnet18_cifar10.yml` (3 seeds × 12 rungs),
`ptq_ladder_vit_s_cifar10.yml` (3 seeds × 7 rungs), `ptq_validation_imagenet.yml` (ResNet-18/50,
paper settings). Re-running a sweep resumes it.

Before interpreting SmoothQuant, measure whether there are outliers to migrate:

```python
from edge_ai_compression.compression.quantization.smoothquant import outlier_stats

outlier_stats(model, train_loader)  # max/median of per-channel |activation| maxima per layer
```

## Deviations from the papers (documented, deliberate)

- **Iterations:** AdaRound 2k per layer and BRECQ 5k per block on CIFAR (the papers used 10k–20k
  on ImageNet). The ImageNet validation sweep uses the papers' counts.
- **BRECQ:** block-wise rounding only; activation step sizes are calibrated, not learned
  jointly.
- **QAT:** BN stays folded (D3.7); short fine-tunes from the PTQ model.
- **Activations:** symmetric int8 including post-ReLU tensors, matching the kernels. This wastes
  one bit on non-negative activations; unsigned activations are a possible extension.
- **Validation against paper numbers:** needs ImageNet (license-gated). Compare
  `ptq_validation_imagenet` results with the ImageNet tables of Nagel et al. 2020 and Li et al.
  2021.

## Results

- ResNet-18 / CIFAR-10 ladder: [PENDING: run `configs/sweeps/ptq_ladder_resnet18_cifar10.yml`]
- ViT-S / CIFAR-10 SmoothQuant track: [PENDING: run `configs/sweeps/ptq_ladder_vit_s_cifar10.yml`]
- ImageNet validation: [PENDING: run `configs/sweeps/ptq_validation_imagenet.yml` (needs ImageNet)]
