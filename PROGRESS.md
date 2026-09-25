# PROGRESS

Resume point for autonomous work. Updated after every commit.

- **Mission:** see the mission brief (pre-training → post-training/compression → export → C++ inference → benchmark → analysis).
- **Decisions:** `docs/DECISIONS.md`. **Interview prep:** `docs/INTERVIEW_PREP.md`.
- **Branch stack:** `main` ← `consolidate` (Phase 0) ← `phase1-benchmarking` ← `phase2-kernels` ← ...
- **Environment:** `.venv/` (Python 3.12 locally, CI 3.11). Checks before every commit:
  `ruff check . && ruff format --check . && pytest -q`.

## Current phase: 1 (benchmarking harness), branch `phase1-benchmarking`

### Done
- Phase 0 complete on `consolidate` (pushed). PR not opened: `gh` token lacks PR permission (see NEEDS ESHAAN).
- `benchmarking/stats.py`: summarize, bootstrap_ci, compare (MWU, Cliff's delta, ratio CI).
- `benchmarking/config.py`: BenchmarkConfig with strict key validation.
- `benchmarking/timing.py`: measure_latency (warmup/timed phases, GC off, injectable clock), Linux CPU affinity.
- `benchmarking/fingerprint.py`: static (hashed) / dynamic / git.
- Ruff pinned to 0.16.8 in pre-commit and the dev extra.

### Next (Phase 1)
1. `benchmarking/memory.py` + `benchmarking/worker.py` + `benchmarking/isolation.py` (spawned-process benchmark, cold-start stages). Config: replace `input_shape` with `batch_size` (D1.7).
2. `benchmarking/energy.py` (EnergyMeter protocol, NullMeter only).
3. `benchmarking/report.py` + rewrite `evaluator.py` on top of the above.
4. Experiment DB: new schema, header-mismatch error, results_dir parameter; always-on in the runner; delete ResultLogger / runs.jsonl / benchmark_append.md; surrogate + search_compression column renames.
5. Delete `hardware/`, `experiments/run_hardware_eval.py`, `configs/hardware/`, `latency_profiler.py`, `memory_profiler.py`, `energy_estimator.py`, `configs/experiments/with_experiment_db.yml`.
6. `benchmarking/cli.py` + `configs/experiments/smoke_benchmark.yml`; update the other configs.
7. README + docs; INTERVIEW_PREP Phase 1; push branch.

### Later phases
2 kernels + export · 3 PTQ/QAT ladder · 4 pruning + recovery · 5 pre-training + signals · 6 studies + analysis · 7 paper/blog · 8 small-LM (stretch)

## Open issues
- The laptop was on battery during development: real benchmark runs need AC power (strict_environment).

## NEEDS ESHAAN
1. **Open the Phase 0 PR.** `gh` token can't create PRs. Run `! gh auth login -h github.com -w`, then:
   `gh pr create --base main --head consolidate --title "Consolidate legacy code into legacy/"`.
   Later phase PRs stack: `phase1-benchmarking` → base `consolidate`, and so on.
