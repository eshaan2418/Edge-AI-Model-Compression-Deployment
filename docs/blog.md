# Can you tell early whether a model will compress well?

*A plain-language tour of this project: what it asks, what we built, and what measuring
carefully taught us. Results marked PENDING are filled in only from the experiment database.*

## The question

To run a neural network on a phone, a laptop CPU, or a small board, you usually
**compress** it:

- **quantize** it: store weights, and often activations, as 8- or 4-bit integers instead of
  32-bit floats
- **prune** it: remove weights or whole channels

Some trained models survive this with almost no accuracy loss, and others fall apart. The
usual workflow finds out only at the very end: train for days, compress, measure, discover the
model is fragile, repeat.

This project asks two connected questions:

1. **Can training itself make models more compressible?** For example, by injecting
   quantization noise while training, penalizing heavy-tailed weight distributions, or training
   sparse from the start.
2. **Can you tell early in training how compressible the final model will be?** If a few cheap
   measurements a tenth of the way through training predict the final model's accuracy after
   4-bit quantization, you can stop bad runs early.

## What "cheap measurements" means

While a model trains, we periodically measure a handful of numbers on a fixed batch of
training images:

- **How heavy-tailed are the weights?** (kurtosis) A few huge weights force a coarse
  quantization grid for everyone else.
- **How extreme are the activations?** A few enormous activation channels waste an 8-bit range.
- **How curved is the loss surface?** (the trace of the Hessian) Sharp valleys punish small
  weight changes, and quantization *is* a small weight change.
- **How much does the loss rise if you nudge the weights in the worst direction?** (sharpness)

Then we train many models: different sizes, three random seeds each, and several
"compression-aware" training recipes. We compress every final model with the same fixed set
of methods and ask a simple predictor to guess each model's accuracy drop from the early
measurements.

The part that makes this honest is how the predictor is scored. It is only ever tested on a
*kind* of training run it has never seen: all seeds of a configuration are held out together.
Otherwise it could score well just by recognizing near-duplicates. It also has to beat a
baseline that only knows the model's size. "Bigger models compress better" is not a
discovery.

**Result:** [PENDING: run the Phase 5 tracks and `compress_runs`, then `./reproduce.sh`; this
section is written from `results/figures/early_prediction_*.csv` only]

## Measuring speed without fooling yourself

The second half of the project is about *speed*, and speed is easy to mismeasure.

**Time one process many times and you learn about that one process.** Timing a model
1,000 times in a row gives you 1,000 numbers that aren't independent. The caches are warm, the
memory layout is fixed, and the processor settles into one state. Confidence intervals
computed from them come out far too narrow. We treat a fresh process as the unit of
measurement: several processes, each timed many times. Our uncertainty comes from
how much the processes disagree with each other.

**FLOPs are not time.** Papers often report floating-point operation counts as an efficiency
proxy. But a sparse matrix kernel doing half the arithmetic can easily be *slower* than a
dense one, because each nonzero weight needs an index lookup and an irregular memory access.
We wrote our own CPU kernels (dense, 8-bit, 4-bit, and two kinds of sparse) and measured where
each one sits relative to the hardware's limits.

**Result:** [PENDING: run the kernel study and backend sweep on the M5 Pro, then `./reproduce.sh`]

## Things that were silently wrong

Building the measurement pipeline turned up bugs that would each have quietly corrupted
results. None of them raised an error:

- **Memory:** on Linux, a freshly started worker process reported its *parent's* peak memory
  as its own. The operating system carries that counter across process start-up.
- **Rounding:** our kernel quantized by multiplying by 1/scale while the reference divided by
  the scale. The two differ in the last bit, which flips values sitting exactly on a
  rounding boundary, and those flips add up through a deep network.
- **FLOPs:** our FLOP counter recognized only ordinary layer types, so every quantized model
  counted as zero FLOPs.
- **Quantization:** PyTorch's built-in quantization did not run at all on the primary laptop,
  because no quantized backend was selected by default.
- **Checkpoints:** a typo in a checkpoint path made the pipeline silently benchmark a
  randomly initialized model.

Each is now covered by a test. The lesson generalizes: for any number you plan to publish,
ask what would have to be true for it to be wrong, then test that.

## What's in the repository

- A benchmarking harness with fresh-process repetition, bootstrap confidence intervals, and a
  hardware/software fingerprint on every measurement.
- C++ kernels with a small engine that runs ResNets on them, plus ONNX Runtime and PyTorch
  baselines.
- A quantization ladder (from simple rounding to AdaRound, BRECQ, quantization-aware
  fine-tuning, and Hessian-guided mixed precision) and a pruning ladder with low-rank recovery
  that preserves sparsity.
- Compression-aware training variants and signal logging.
- Analyses (early prediction, scaling curves, latency proxies, Pareto frontiers). Each one is
  tested on synthetic data where the right answer is known in advance.

Every figure regenerates from the experiment database with one command: `./reproduce.sh`.
