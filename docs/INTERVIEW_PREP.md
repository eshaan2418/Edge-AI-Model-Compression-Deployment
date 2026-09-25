# Interview prep

Five hard questions per phase, with concise answers.

---

## Phase 0: consolidation

**1. Why move the old code to `legacy/` instead of deleting it? When would you delete it?**
The TF/TFLite scripts are the only working int8 TFLite export path and a reference for the Phase 2 export work. Keeping them frozen and lint-checked costs nothing. Delete once the unified PyTorch → ONNX export path reproduces what they do.

**2. Why drop the `mc` console script instead of packaging `legacy`?**
Packaging it would install a top-level module named `legacy` (previously `src`) into site-packages, where it can shadow or collide with other packages. Legacy code is reference material, not a library, so it runs from the repo root.

**3. The framework's quantization only converts `nn.Linear`. What does that mean for the existing ResNet-18 results?**
ResNet-18 has one Linear layer (the classifier). Nearly all compute is in convolutions, which stay FP32, so "quantized ResNet-18" results so far are close to a no-op. Static per-channel conv quantization is a prerequisite for any quantization claim (Phase 3).

**4. What does the CI smoke experiment catch that unit tests don't, and what can't it catch?**
It catches integration breaks: config parsing, data loading, the compression pipeline, evaluation and DB logging all wired together. It can't catch numerical correctness or performance regressions, because it uses random data and tiny iteration counts on a shared runner.

**5. How do you make sure every number in the README traces to a run?**
The experiment DB is the only results output. Every row has an experiment ID, and artifacts (config, metrics, fingerprint, traces) are stored under that ID. Figures are generated from the DB by a script. No hand-typed numbers; missing results are marked `[PENDING: run <config>]`.

---

## Phase 1: benchmarking harness

**1. Why is a process, not an iteration, your unit of replication? What goes wrong otherwise?**
Iterations in one process share cache state, allocator layout, JIT/dispatch warm state and CPU frequency, so they're autocorrelated. A bootstrap over 1000 iterations acts as if you had 1000 independent samples and gives a CI that's far too narrow. Much of the real variance is between processes (address-space layout, scheduling, which cores the OS picks). Kalibera & Jones (2013) formalize this as hierarchical repetition. I report pooled per-iteration percentiles only as descriptive distributions, and CIs and tests only over per-process medians.

**2. With 5 processes, how much should anyone trust your bootstrap CI?**
Only somewhat. The percentile bootstrap under-covers at n=5, and the smallest two-sided Mann-Whitney p with 5 vs 5 is about 0.008. So n=5 detects large effects but gives indicative intervals. For headline claims I'd raise `process_repeats` to 10 or more, or report an exact order-statistic interval for the median. The docs and DECISIONS D1.1 say this explicitly.

**3. You can't pin threads on macOS. How do you get defensible latency numbers on an M5 Pro?**
Control what I can and record what I can't. Fix the thread count below the Super-core count, refuse to run on battery or in Low Power Mode, disable GC during timing, run fresh processes, record load average and thermal notes in the fingerprint, and label numbers "OS-scheduled". For A-vs-B claims the remaining scheduler noise shows up as between-process variance, which the CI already includes. On Linux (Colab/CI) I can pin with `sched_setaffinity`.

**4. How do you measure peak memory, and why not "memory used by inference"?**
`ru_maxrss` gives the process high-water mark, but it can't be reset on macOS. Checkpoint loading also creates a transient buffer that can exceed the inference peak. So an "inference-only" delta isn't cleanly measurable, and a background RSS sampler misses short spikes. I report two well-defined numbers: the whole-process peak (what you'd need to deploy) and the peak above the torch runtime baseline (what this model adds, including load transients).

**5. What did building the harness uncover about the existing code?**
Four things:
- The old "peak RAM" was the whole pytest/Python process after importing torch (about 1.2 GB), not the model's footprint.
- "Energy" was a linear formula of latency and size, and it was a surrogate training target.
- On the primary platform, dynamic quantization didn't run at all: torch 2.14 on arm64 has no default quantized engine (`NoQEngine`).
- A missing checkpoint path silently benchmarked random weights.

All four are fixed and tested. The quantization finding also changes Phase 3's design, because torch has deprecated its quantized tensor dtypes.

---

## Phase 2: C++ kernels + export

**1. Your int8 GEMM on x86 doesn't use `_mm256_maddubs_epi16`, the standard fast path. Why not?**
`maddubs` multiplies unsigned by signed bytes and adds adjacent pairs into saturating int16. With symmetric int8 activations you'd have to shift them to unsigned (then correct with a row-sum term), and two products of about 128×127 already overflow int16. So it isn't exact. I sign-extend both operands to int16 and use `madd_epi16`, which sums pairs into int32 and is exact. That costs about 2× the multiply throughput of `maddubs`. VNNI (`vpdpbusd`) fixes the saturation but still needs unsigned activations. A test with K=4099 and all -128 operands would catch a saturating path.

**2. Walk me through the NEON int8 microkernel.**
`vdotq_laneq_s32(acc, b, a, lane)` adds, to each of 4 int32 lanes i, the dot product of 4 bytes of `b` (lane i) with 4 bytes of `a` selected by `lane`. I pack activations as 4 columns × 4 k-values per 16-byte vector and weights as 4 rows × 4 k-values per vector. With lane = row r, one instruction produces 4 output columns for row r. A 4×16 tile uses 16 accumulator registers, 4 B vectors and 1 A vector: 21 of 32 NEON registers, 16 dot instructions (256 multiply-adds) per 80 bytes loaded. Output rows are contiguous, so stores need no transpose.

**3. Why does 50% unstructured sparsity rarely beat dense on a CPU, and why can 2:4 do better?**
Dense GEMM reuses each loaded activation across a whole register tile: one B load feeds four FMAs, and weights are broadcast. A sparse kernel has to fetch the activation row that matches each nonzero, so it does one load per FMA vector plus index decoding, and it loses register blocking across rows. At 50% sparsity you save half the FLOPs but become load-bound, and CSR indices double the bytes per weight (a 50% CSR matrix is exactly as big as dense fp32; I measured this). 2:4 caps the metadata at 2 bits per value and makes every group the same shape, which is what the hardware sparse tensor cores exploit. On a CPU without such units the gain has to come from halved loads, so the crossover is an empirical question. My kernel study measures it per layer shape.

**4. How do you know your roofline analysis isn't flattering your kernels?**
Two choices make it conservative. Arithmetic intensity uses compulsory traffic (each operand moved once), which overstates AI, so "efficiency vs attainable" is measured against an optimistic roof. And the peaks come from single-core microbenchmarks on the same machine (independent FMA / dot chains, STREAM-style triad), logged with the same fingerprint as the kernel rows, so I never compare against spec-sheet numbers. The analysis refuses to mix machines.

**5. How does a benchmark of your C++ engine stay comparable with PyTorch and ONNX Runtime?**
Every backend goes through the same path:
- Export one artifact, then time it in fresh processes with the same warmup/iteration protocol.
- The same input shape, 1 thread for all (my kernels are single-threaded; ORT gets `intra_op_num_threads=1`).
- Accuracy measured with the backend itself.
- `size_mb` = the artifact on disk. The engine stores compressed payloads, so int8/int4 size is real.
- Cold start includes loading and packing weights, as it would for a real runtime.

Two bugs this caught before any results were produced: the dynamo ONNX exporter writing weights to a side file (so size looked like 0.1 MB), and Linux `ru_maxrss` inheriting the parent's peak across exec.

---

## Phase 3: PTQ / QAT ladder

**1. Why build quantization on fake-quant simulation instead of PyTorch's quantized modules?**
Three reasons:
- On the primary platform, torch 2.14's eager quantization didn't even run (no quantized engine), and torch has deprecated its quantized dtypes.
- Simulation is differentiable, which AdaRound, BRECQ and LSQ all need.
- Simulation is backend-independent: the deployed model is the C++ engine, not torch.

The risk is that simulation and deployment silently disagree. So every lowered layer is tested against its simulation (1e-5 relative, identical inputs), and accuracy is always measured with the backend that's timed.

**2. Your engine and simulation disagree by about 1% at the logits. Isn't that a bug?**
It was 1.6% at first, and part of it was a bug: the kernel multiplied by 1/scale while the simulation divides, which flips rounding ties. After that fix, the rest is intrinsic. The engine accumulates exactly in int32 and rescales once; the simulation accumulates dequantized products in fp32. They differ around 1e-7, and a value sitting exactly on a rounding boundary at the next layer can round differently. Those one-step flips compound through 20 layers of a random-weight network. Any two correct int8 implementations disagree this way. The right contract is per-layer exactness plus end-to-end agreement, and that's what I test.

**3. What does AdaRound optimize, and why does it beat round-to-nearest?**
Nearest rounding minimizes each weight's own error but ignores how errors combine in the layer output. AdaRound learns a rounding direction per weight so that the layer's output on real data, ‖WX − W̃X‖², is minimized. It relaxes the binary choice to a rectified sigmoid, and an annealed regularizer (β from 20 to 2) pushes it back to 0/1. A second-order Taylor expansion shows the output error is what drives the task loss, so correlated rounding choices can cancel each other. My unit test reproduces the mechanism: at 3 bits, AdaRound's reconstruction error is lower than RTN's on the same layer.

**4. How does HAWQ decide which layers get 4 bits?**
- **Sensitivity:** each layer gets Ω(b) = tr(H)/n · ‖Q_b(W) − W‖², the average loss curvature for that layer times the size of the quantization perturbation.
- **Curvature:** Hutchinson's estimator, tr(H) = E[vᵀHv] with Rademacher v. One double-backprop Hessian-vector product gives the trace of every layer's diagonal block at once.
- **Allocation:** an exact ILP (scipy milp) picks bits per layer to minimize total Ω under an average-bit budget, with first and last layers pinned to 8.

Caveats: the metric treats layers independently (no cross-layer interaction terms), and the trace is taken at the full-precision weights.

**5. Why would SmoothQuant fail to help on your ViTs, and how would you know?**
SmoothQuant moves per-channel activation scale into the weights. That only helps if a few activation channels are much larger than the rest, so per-tensor activation scales waste resolution. Dettmers et al. saw systematic outlier features emerge around the billion-parameter scale, and a small CIFAR ViT may have none. So I measure first: `outlier_stats` reports max/median per-channel activation maxima at each LayerNorm-fed linear. If the ratios are near 1–10, I expect SmoothQuant ≈ RTN and report that as a negative result. A test with an injected outlier channel confirms the method works when the precondition holds.

---

## Phase 4: pruning + recovery

**1. Why does channel pruning only remove channels inside residual blocks?**
A residual block computes x + F(x), so F's output channels must match x's. Removing an output channel of the block's last conv would require removing the same channel from the skip path and every block that reads it: coupled groups across the network (the problem DepGraph, Fang et al. 2023, solves in general). The inner channels (conv1's outputs, which are only read by conv2) are free to remove without touching anything else. It's a clean, always-valid subset. The limitation is that block outputs stay at full width.

**2. You recover sparse models with LoRA. What goes wrong if you merge the adapters naively?**
The merged weight is W + BA, and BA is a dense matrix, so every pruned zero becomes nonzero and the model is dense again. The sparsity win disappears without any error being raised. I mask the update during training *and* at merge time (W + M ⊙ BA), so the network trains in exactly the function space it will be deployed in. A unit test checks that every pruned weight is still zero after recovery.

**3. What's the difference between `pruning_sparsity` and `weight_sparsity`, and why store both?**
`pruning_sparsity` is what I asked for; `weight_sparsity` is what the evaluated model actually has. They diverge for real reasons:
- 2:4 skips the stem (K = 27 isn't divisible by 4).
- Channel pruning physically removes weights instead of zeroing them.
- Quantization can round small weights to zero.

Any "sparsity vs latency" analysis has to use the measured number, or it attributes speedups to sparsity that isn't there.

**4. Why would 2:4 beat 50% unstructured sparsity in latency, at the same number of zeros?**
2:4 has fixed, tiny metadata (2 bits per kept value) and every group has the same shape. So a kernel can process groups with no data-dependent branching and predictable loads, and dedicated hardware (Ampere sparse tensor cores) can run it at 2× throughput. Unstructured sparsity at 50% needs a 4-byte index per nonzero: a CSR matrix is as large as the dense one, and each nonzero triggers an indirect load. On a CPU without sparse units, the Phase 2 kernel study measures whether 2:4's regularity is enough to win. That's an empirical claim I don't assume.

**5. How do you make "fine-tune vs LoRA recovery" a fair comparison?**
Both get the same budget: the same epochs, learning-rate schedule and data, starting from the same pruned weights, and the same 3 training seeds. The result is a trade-off, not a single winner: LoRA trains about 1–5% of the parameters (so it fits in less memory and each step's weight update is cheaper), while full fine-tuning can move every surviving weight. I report accuracy recovered per trainable parameter as well as absolute accuracy.

---

## Phase 5: pre-training variants + signals

**1. What does Quant-Noise do that QAT doesn't, and why is it off in eval mode?**
QAT fake-quantizes every weight on every forward pass. Quant-Noise quantizes a random subset (fraction p), so gradients through the unquantized weights stay unbiased, while the network still learns to tolerate quantization error. p = 1 recovers QAT from scratch, so one knob spans the family. It's switched off in eval because signals, checkpoints and downstream PTQ should see the clean float weights. The question is whether training *with* noise yields weights that quantize better *afterwards*, not what the noisy network's accuracy is.

**2. RigL needs the gradient of weights that are currently zero. How do you get it, and what breaks if you use a mask parametrization?**
If the forward pass computes W ⊙ M, autograd gives ∂L/∂W ⊙ M, which is zero at every inactive position, so there's nothing to grow from. I run the forward pass on the dense weight tensor, where inactive entries are simply zero. Then ∂L/∂W is dense, and the growth step can pick the inactive connections with the largest gradient magnitude. The cost is that the optimizer also updates inactive weights, so I re-zero them, and their momentum/Adam state, after every step. A test checks that zeros, per-layer density and momentum masking all hold.

**3. Why measure signals on a training batch rather than the test set?**
Phase 6 uses these signals to predict final post-compression *test* accuracy. Signals computed from test data, especially the loss-based ones (Hessian trace, sharpness, probe loss), would leak test information into the predictor's features and inflate its apparent quality. So a probe batch is drawn once from the training data and frozen, which keeps the signals comparable across steps and runs.

**4. Sharpness here is L(w + ρ g/‖g‖) − L(w). What does it approximate, and what are its limitations?**
It's the first-order solution of SAM's inner problem, max over ‖ε‖ ≤ ρ of L(w + ε), which is ε* ≈ ρ·g/‖g‖. It's cheap (one extra forward and backward pass) and bounds how much the loss can rise under a weight perturbation of size ρ. That's a direct proxy for quantization noise, which is a weight perturbation. Limitations:
- It isn't invariant to layer-wise rescaling of ReLU networks (Dinh et al. 2017), and BN makes that worse.
- First order only.
- A single ρ.

That's why I also log the Hessian trace and weight kurtosis, and Phase 6's ablations measure which signal actually carries predictive information.

**5. How do you scale "training length" properly, and why not just use intermediate checkpoints?**
An intermediate checkpoint of a 90-epoch cosine run at epoch 30 has a high learning rate and hasn't annealed. It isn't the model you'd get by training for 30 epochs. So the length sweep trains separate 10, 30 and 90-epoch runs, each with its own full schedule. Intermediate checkpoints serve a different purpose: the early-prediction features, where "what does this run look like at step k" is exactly the question.

---

## Phase 6: studies + analysis

**1. Why leave-one-configuration-out instead of ordinary cross-validation for the early-prediction study?**
Seeds of the same configuration (model × variant × epochs) are near-duplicates. Random K-fold would put two seeds of one configuration on opposite sides of the split, and the predictor would score well by recognizing the configuration, not by reading the signals. Holding out every seed of a configuration asks the real question: given the signals of a training run on a configuration you've never seen, can you predict how compressible its final model will be? The CIs come from a cluster bootstrap over configurations for the same reason.

**2. Your predictor gets Spearman 0.8. How do you know it isn't just "bigger models compress better"?**
Two baselines are built in. A mean predictor measures skill against nothing, and a params-only ridge regression measures whether the signals add information beyond model size. The claim is only interesting if the signal-based predictor beats params-only, with non-overlapping CIs. The ablations then name which signal group carries the gain. If params-only matches the full model, that's the result, and I'd report it as a negative.

**3. How would you know if an analysis pipeline, rather than the data, produced a finding?**
Every analysis is tested on a synthetic experiment DB with a planted answer: a known signal-to-target relation, a known power-law exponent, known per-kernel costs. The test asserts the pipeline recovers the planted value and doesn't invent structure (a flat series must not beat the constant model). That caught real bugs:
- an MLP predictor mis-specified for the installed scikit-learn version
- an all-empty ID column parsed as float that broke a join

Both were found before any real data existed. CI runs the whole `reproduce.sh` on the real-schema smoke DB.

**4. When does a power-law fit say nothing?**
With few points (I require at least 4 distinct sizes for a 3-parameter fit), heavy seed noise, or no real trend. So each fit carries a seed-bootstrap CI on the exponent and an AICc comparison against a constant model. If the power law doesn't beat the constant, the table says `beats_constant = False`. Extrapolating scaling curves beyond the measured range isn't something this data supports, and I don't do it.

**5. FLOPs is the standard efficiency metric in papers. Why distrust it?**
FLOPs counts arithmetic, not time. Kernels achieve very different FLOP/s: dense int8 on dot-product instructions, sparse CSR bound by indirect loads, 2:4 in between. Non-GEMM work (im2col, requantization, residual adds) also doesn't appear in FLOPs at all. Within one kernel family FLOPs often ranks models fine; across families it can rank a slower model as faster. The study measures exactly this (rank correlation per backend versus pooled) and fits a learned proxy that knows about each kernel's cost curve, evaluated on held-out configurations.
