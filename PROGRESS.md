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

### Phase 2 progress (branch `phase2-kernels`, pushed)
Done:
- `csrc/` C++ kernels + nanobind `_C` (fp32, int8 exact, int4 weight-only, 2:4, CSR, im2col,
  quantize, microbenchmarks); runtime ISA dispatch scalar/NEON/AVX2/AVX-512. Build:
  `pip install -e ".[kernels,export]" && scripts/build_kernels.sh`.
- `inference/packing.py` (numpy formats + references), `inference/kernels.py` (wrappers),
  `inference/engine.py` (FX lowering, BN fold, ReLU fusion, 5 modes, compressed payloads),
  `inference/backends.py` (torch_eager, onnxruntime, edge_{f32,int8,w4,sparse24,csr}).
- Benchmark worker/evaluator are backend-aware; DB schema v3 adds `backend`.
- CI matrix ubuntu (AVX2) + macos-14 (NEON), kernels required. CI runs on every branch push.
- Fixes found via CI: Linux ru_maxrss inherits the parent's peak across exec -> VmHWM (D1.15).

Next:
1. Confirm CI green on both OSes (last failure was the ru_maxrss issue, fix pushed in b2e8317).
2. Roofline: `inference/roofline.py` (machine peaks via microbench, per-layer FLOPs/bytes/achieved).
3. Kernel benchmark table in the experiment DB + fresh-process kernel worker.
4. Sparsity study config + runner (dense vs CSR vs 2:4 across sparsity, ResNet-18 layer shapes).
5. AVX-512 correctness in CI via Intel SDE (if downloadable), else document.
6. Docs (docs/inference.md), INTERVIEW_PREP Phase 2, push.

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
