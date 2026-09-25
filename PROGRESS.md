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

### Next (Phase 2)
Write the Phase 2 plan into DECISIONS (no approval gate), then implement: C++ kernel library
(NEON primary; AVX2/AVX-512 correctness in CI), nanobind/pybind11 bindings, CMake in CI, export path.

### Later phases
3 PTQ/QAT ladder · 4 pruning + recovery · 5 pre-training + signals · 6 studies + analysis · 7 paper/blog · 8 small-LM (stretch)

## Open issues
- The laptop was on battery during development: real benchmark runs need AC power (strict_environment).
- torch 2.14 deprecates quantized tensor dtypes (qint8 etc.). Phase 3 should use fake-quant simulation
  plus our own int kernels rather than torch.ao quantized modules (DECISIONS D1.14).
- Percentile-bootstrap CIs with 5 processes under-cover; use >=10 for headline claims.

## NEEDS ESHAAN
1. **Open the Phase 0 PR.** `gh` token can't create PRs. Run `! gh auth login -h github.com -w`, then:
   `gh pr create --base main --head consolidate --title "Consolidate legacy code into legacy/"`.
   Later phase PRs stack: `phase1-benchmarking` → base `consolidate`, and so on.
