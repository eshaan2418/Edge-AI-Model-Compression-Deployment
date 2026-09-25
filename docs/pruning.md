# Pruning and recovery

Design rationale: [`DECISIONS.md`](DECISIONS.md) (Phase 4, D4.x). No results are quoted here.

## Modes (`compression.pruning.mode`)

| Mode | What is removed | Structure | Kernel it runs on |
|---|---|---|---|
| `global_unstructured` | the `amount` fraction of smallest-magnitude conv/linear weights, globally | none (irregular zeros) | `edge_csr` |
| `layerwise_adaptive` | per-layer sparsities from a scorer (magnitude / gradient / activation / ablation) | none | `edge_csr` |
| `nm` | all but the `n` largest of every `m` consecutive weights along [in·kh·kw] (default 2:4); layers with K % m ≠ 0 stay dense | semi-structured | `edge_sparse24` (2:4) |
| `channel` | the `amount` fraction of each residual block's inner channels, by L1 filter norm or \|BN γ\| (`criterion`) | structured: the model is rebuilt smaller | any dense backend |

## Recovery (`compression.pruning.recovery`)

| `method` | Trains | Keeps the sparsity pattern by |
|---|---|---|
| `none` | nothing | — |
| `finetune` | all parameters | the `torch.nn.utils.prune` mask reparametrization (weight = weight_orig ⊙ mask) |
| `lora` | rank-`rank` adapters per conv/linear + biases; pruned weights frozen | masking the update: W + M ⊙ (α/r)·BA, merged at the end |

Options: `epochs` or `max_steps`, `lr`, `weight_decay`, `rank`, `lora_alpha`. An *unmasked*
LoRA merge would add a dense low-rank matrix to a sparse layer and silently densify it. Masking
the update is what makes LoRA usable for sparse recovery.

## Measured vs configured sparsity

`pruning_sparsity` in the DB is the configured target. `weight_sparsity` is measured on the
evaluated model: the zero fraction of all conv/linear weights, including layers the method
skipped (the 2:4 stem with K = 27) and after quantization. Analyses use `weight_sparsity`.

## Running

```bash
python run_experiment.py --config edge_ai_compression/configs/experiments/smoke_prune.yml   # 2:4 + LoRA
python -m edge_ai_compression.experiments.run_track --track resnet18_prune --device cuda     # full ladder
```

The ladder (`configs/sweeps/prune_ladder_resnet18_cifar10.yml`) is 3 seeds × 13 rungs:
unstructured 50/90%, 2:4, channel 25/50%, each with no recovery, masked fine-tuning, and masked
LoRA at equal budgets (5 epochs, lr 0.01).

## Results

- Pruning ladder: [PENDING: run `configs/sweeps/prune_ladder_resnet18_cifar10.yml`]
- Latency of each structure on its kernel: from the same runs (backend column), plus the Phase 2
  kernel study for the kernel-level crossover.
