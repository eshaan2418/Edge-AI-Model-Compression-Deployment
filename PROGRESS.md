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

### Next: Phase 3 (PTQ/QAT ladder), branch `phase3-ptq` stacked on `phase2-kernels`
Plan to write into DECISIONS first. Key constraint (D1.14): torch 2.14 deprecates quantized dtypes,
so build the ladder on fake-quant simulation + our int kernels, not torch.ao quantized modules.

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
