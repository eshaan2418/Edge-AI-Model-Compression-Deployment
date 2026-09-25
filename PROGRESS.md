# PROGRESS

Resume point for autonomous work. Updated after every commit.

- **Mission:** see the mission brief (pre-training → post-training/compression → export → C++ inference → benchmark → analysis).
- **Decisions:** `docs/DECISIONS.md`. **Interview prep:** `docs/INTERVIEW_PREP.md`.
- **Branch stack:** `main` ← `consolidate` (Phase 0) ← `phase1-benchmarking` ← `phase2-kernels` ← ...
- **Environment:** `.venv/` (Python 3.12 locally, CI 3.11). Checks before every commit:
  `ruff check . && ruff format --check . && pytest -q`.

## Current phase: 2 (C++ kernels + export), branch `phase2-kernels` (stacked on `phase1-benchmarking`)

### Phase 1 (benchmarking harness): COMPLETE, pushed on `phase1-benchmarking`
- `benchmarking/`: stats (bootstrap CI, MWU, Cliff's delta), config (strict keys), timing (GC off,
  warmup/timed bounds), fingerprint (static hashed / dynamic / git), memory (ru_maxrss), worker +
  isolation (fresh process per repeat, cold-start stages), energy (NullMeter), report, evaluator.
- Experiment DB schema v2, always on, the only output. Header mismatch raises. results_dir configurable.
- Removed: markdown table, runs.jsonl, hardware/, latency_profiler, memory_profiler, energy_estimator,
  with_experiment_db.yml, configs/hardware/.
- Fixes: dynamic quantization NoQEngine on arm64; silent random-weights fallback on missing
  checkpoint; compression_order column recorded stages that didn't run.
- Configs: smoke_cpu.yml, smoke_benchmark.yml (both in CI). Docs: docs/benchmarking.md.

### Phase 2 (C++ kernels + export): COMPLETE, pushed on `phase2-kernels`
- `csrc/` kernels + nanobind `_C`; `inference/` packing, kernels, engine (FX, BN fold, ReLU fusion,
  5 modes, compressed payloads), backends (torch_eager, onnxruntime, edge_*), kernel_bench +
  kernel_worker, roofline; `analysis/kernel_study.py`; `run_kernel_study.py`.
- DB schema v3 (`backend` column) + `kernel_benchmarks.csv`.
- CI: ubuntu (AVX2) + macos-14 (NEON), kernels required; optional SDE AVX-512 job (gated).
- Research configs ready, results PENDING (need AC power): `studies/kernel_sparsity_m5.yml`,
  `sweeps/backend_latency_m5.yml`. Docs: docs/inference.md.

### Phase 3 (PTQ/QAT ladder): COMPLETE, pushed on `phase3-ptq`
- Trainer (`pretraining/`, SGD|AdamW, log-spaced checkpoints, export_path, training_runs.csv).
- Ladder: rtn (+min-max/percentile/MSE calibration), adaround, brecq (+Fisher), qat (LSQ), hawq
  (Hutchinson + ILP), smoothquant (+ outlier_stats); ViT t/s/m; ImageNet loader + torchvision
  pretrained models for paper validation.
- Engine `edge_quant`: per-layer exact lowering (D3.6); torch.ao removed (D3.5).
- Resumable grid sweeps (`axes:`), dotted `--set` overrides, DB merge, `run_track` (smoke-tested),
  `notebooks/tracks.ipynb`. Docs: docs/quantization.md.
- Results PENDING (heavy runs, see NEEDS ESHAAN 4-6).

### Phase 4 (pruning + recovery): COMPLETE, pushed on `phase4-pruning`
- `compression/pruning/`: nm (2:4), channel (inner residual channels, L1 | BN gamma), recovery
  (masked fine-tune, masked LoRA), weight_sparsity. PruningSection modes/recovery/tag. DB schema v4.
- `run_track.py` (generic tracks: resnet18_ptq, vit_s_ptq, resnet18_prune), `notebooks/tracks.ipynb`.
- Ladder: `sweeps/prune_ladder_resnet18_cifar10.yml` (39 runs). Docs: docs/pruning.md.
- Results PENDING (NEEDS ESHAAN 7).

### Phase 5 (pre-training variants + signals): COMPLETE, pushed on `phase5-pretraining`
- ResNet family resnet{10,18,34}_w{0.25,0.5,1.0}_cifar; signals.py (kurtosis, outliers, Hessian
  trace, sharpness, ...) logged to training_signals.csv; variants quant_noise / kurtosis / rigl;
  training sweeps (`kind: train`, path templates); tracks pretrain_variants/scaling/length/vit.
- Fixes: circular import (also patched on phase4-pruning, 6bab7ac), strict experiment keys (D5.3).
- Docs: docs/pretraining.md. Results PENDING (NEEDS ESHAAN 8).

### Phase 6 (studies + analysis): COMPLETE (code), pushed on `phase6-studies`
- compress_runs (panel -> compressibility targets, schema v5 source_run_id/source_step);
  analysis: early_prediction (LOCO CV, cluster-bootstrap CIs, ablations), scaling (power law + AICc),
  latency_proxy (naive + learned kernel model), pareto_report; reproduce.sh (+ CI) with MANIFEST.
- Fixes: FLOPs = 0 for quantized models (D6.3), MLP predictor arg, float-parsed empty id column.
- Docs: docs/analysis.md. All results PENDING on the heavy runs (NEEDS ESHAAN 3-9).

### Next: Phase 7 (paper + blog), branch `phase7-paper`
paper/ (LaTeX, workshop length) + docs/blog.md + README headline, all with [PENDING] markers
where results don't exist; figures only via reproduce.sh.

### Later phases
3 PTQ/QAT ladder · 4 pruning + recovery · 5 pre-training + signals · 6 studies + analysis · 7 paper/blog · 8 small-LM (stretch)

## Open issues
- The laptop was on battery during development: real benchmark runs need AC power (strict_environment).
- torch 2.14 deprecates quantized tensor dtypes (qint8 etc.). Phase 3 should use fake-quant simulation
  plus our own int kernels rather than torch.ao quantized modules (DECISIONS D1.14).
- Percentile-bootstrap CIs with 5 processes under-cover; use >=10 for headline claims.
- Kernels are single-threaded (D2.2): compare against torch/ORT at 1 thread.

## NEEDS ESHAAN
1. **Open the Phase 0 PR.** `gh` token can't create PRs. Run `! gh auth login -h github.com -w`, then:
   `gh pr create --base main --head consolidate --title "Consolidate legacy code into legacy/"`.
   Later phase PRs stack: `phase1-benchmarking` → base `consolidate`, and so on.
2. **(Optional) Enable AVX-512 correctness tests in CI.** They run the AVX-512 kernels under Intel SDE
   emulation, which requires accepting Intel's SDE license. If you accept it: repo Settings ->
   Secrets and variables -> Actions -> Variables -> New variable `ENABLE_SDE` = `true`.
   Until then AVX-512 code is compiled in CI but not executed (GitHub x86 runners lack AVX-512).
3. **Run the Phase 2 studies on the M5 Pro** (AC power, close other apps, ~idle machine):
   ```bash
   source .venv/bin/activate && scripts/build_kernels.sh
   python run_kernel_study.py --config edge_ai_compression/configs/studies/kernel_sparsity_m5.yml
   python launch_sweep.py --config edge_ai_compression/configs/sweeps/backend_latency_m5.yml
   python -m edge_ai_compression.analysis.kernel_study --results results --study kernel_sparsity_m5
   ```
   Results land in `results/` (experiment DB). The strict environment check refuses to run on battery.
4. **Phase 3 ResNet-18 ladder (GPU, Colab/Kaggle):** open `notebooks/tracks.ipynb`, run all
   cells for the resnet18 track (trains 3 seeds x 60 epochs, then 36 ladder runs). Or locally with
   the M5 GPU: `python -m edge_ai_compression.experiments.run_track --track resnet18_ptq --device mps`.
   Download `track_results.zip` and merge: `python -m edge_ai_compression.experiment_db.merge --src <dir>/results`.
5. **Phase 3 ViT track:** same notebook, vit_s_ptq track (3 seeds x 200 epochs + 21 runs).
6. **(Optional) ImageNet validation vs the papers:** needs ImageNet-1k (license-gated) in
   `data/imagenet/{train,val}` ImageFolder layout, then
   `python launch_sweep.py --config edge_ai_compression/configs/sweeps/ptq_validation_imagenet.yml --set device=cuda`.
7. **Phase 4 pruning ladder:** `notebooks/tracks.ipynb`, resnet18_prune cells (reuses the Phase 3
   ResNet-18 checkpoints; trains them if missing).
8. **Phase 5 training tracks:** `notebooks/tracks.ipynb` cells for pretrain_variants (15 runs),
   pretrain_scaling (30), pretrain_length (9), pretrain_vit (9). These produce the checkpoints and
   signals Phase 6 analyzes.
9. **Phase 6 targets + figures** (after 8): `python -m edge_ai_compression.experiments.compress_runs --device cuda`
   (add `--all-steps` for compressibility-over-training), then `./reproduce.sh`. Check
   `results/figures/MANIFEST.md`.
