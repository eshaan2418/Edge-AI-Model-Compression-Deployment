# Analysis toolkit

Four standalone analyses go beyond point estimates to quantify *how much you can
trust* a measurement, *how far a model is from the hardware limit*, *which layers
matter most*, and *how good a Pareto front is*. Everything runs on CPU, offline,
on real measurements — never fabricated numbers.

## Statistical latency A/B testing

`benchmarking/statistics.py` (library) + `benchmarking/ab_compare.py` (CLI).

A single mean latency hides run-to-run noise. This compares two models with the
tools you'd use to defend a claim in review:

- **Bootstrap confidence intervals** (percentile method) on the median latency
  and on the speedup ratio — a seeded resample, so results are deterministic.
- **Mann–Whitney U** test for a significant difference in distribution.
- **Cliff's delta** effect size (negligible / small / medium / large) so
  "significant" is separated from "large enough to matter".
- A plain-English **verdict** combining the p-value and effect size.

```bash
python -m edge_ai_compression.benchmarking.ab_compare \
    --model-a small_cnn_student --model-b resnet18_cifar --repeats 200
```

> **Honesty caveat.** This is wall-clock latency on the *benchmarking host*, not
> the target edge device. Use it to compare candidates on equal footing, not to
> predict on-device latency.

## Roofline & arithmetic intensity

`theory/roofline.py` (library + CLI), backed by
`theory/model_complexity.py` (`estimate_activation_bytes`, `weight_bytes`) and
the compute/bandwidth ceilings on each `hardware/profiles.py` profile.

The [roofline model](https://en.wikipedia.org/wiki/Roofline_model) (Williams et
al., 2009) places a workload's **arithmetic intensity** (FLOPs per byte moved,
where FLOPs = MACs × 2) against a device's peak compute and memory bandwidth. It
tells you whether a model is **compute-bound** or **memory-bound**, where the
ridge point is, and a **theoretical latency lower bound**:

```bash
python -m edge_ai_compression.theory.roofline \
    --model resnet18_cifar --profile raspberry_pi
```

Pass `--measured-latency-ms` to also report roofline **efficiency**
(lower_bound / measured).

> **Honesty caveats.** The per-profile `peak_gflops` and `mem_bandwidth_gbs` are
> **nominal vendor-spec ceilings**, not measured throughput — planning numbers.
> The reported latency is an explicit *theoretical lower bound*: real latency is
> always ≥ it, so efficiency is ≤ 1.

## Layer-wise sensitivity

`analysis/layer_sensitivity.py` (library + CLI).

To compress non-uniformly you need to know which layers tolerate it. For each
Conv2d / Linear layer independently (on a deep copy), this perturbs *only* that
layer and measures the **numerical output drift** on a fixed seeded batch:

- `--method prune` — zero the smallest-magnitude `--amount` fraction of weights.
- `--method quantize` — per-tensor uniform `--bits`-bit round-trip.

It reports `drift_rel_l2 = ‖y' − y‖ / ‖y‖` and cosine similarity, ranked so the
most sensitive layers surface first:

```bash
python -m edge_ai_compression.analysis.layer_sensitivity \
    --model small_cnn_student --method prune --amount 0.5
```

> **Honesty caveat.** This measures *output drift*, a proxy — **not** accuracy
> loss. A high-drift layer is a candidate to protect, but confirm with a real
> evaluation before committing a per-layer schedule.

## Pareto quality indicators

`optimization/indicators.py` (library + CLI).

Finding the Pareto front tells you *which* trade-offs are non-dominated; it does
not say how *good* the front is. These reduce a front to comparable numbers:

- **Hypervolume** — objective-space volume the front dominates up to a reference
  (nadir) point. Larger is better; rewards convergence and spread at once.
- **Additive epsilon-indicator** — smallest shift for the front to dominate a
  reference front. Smaller is better.
- **Schott spacing** — how evenly the front is distributed. Smaller is more
  uniform.

Objectives use the project keys (maximize `accuracy`; minimize
`latency_ms_mean`, `size_mb`, `peak_ram_mib`), normalized to a common
minimize-form.

```bash
python -m edge_ai_compression.optimization.indicators \
    --results results/experiments.csv --out reports/indicators.json
```

## Property-based tests

`tests/test_properties.py` uses [Hypothesis](https://hypothesis.readthedocs.io/)
to assert *laws* over generated inputs rather than hand-picked examples:
dominance is a strict partial order, bootstrap CIs bracket their point estimate,
Cliff's delta is bounded and antisymmetric, hypervolume ignores dominated points,
`deep_merge` is a non-mutating idempotent merge, and the roofline attainable rate
never exceeds peak. Install with `pip install -e ".[dev]"` (pulls Hypothesis).
