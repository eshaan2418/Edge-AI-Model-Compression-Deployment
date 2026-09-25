# paper/

Workshop-length write-up (`main.tex`, `refs.bib`).

- **Numbers:** every table is `\input` from `generated/*.tex`, and figures come from
  `../results/figures/`. `./reproduce.sh` (run from the repo root on the real DB) regenerates both.
  Anything without results renders as a red **PENDING** box. Never type a number into
  `main.tex` by hand.
- **Headline finding** (abstract): write it only after the Phase 5/6 runs, from
  `results/figures/`, and cite the run IDs in the experiment DB.
- **References:** entries in `refs.bib` were written from memory of the papers' titles, venues
  and authors. Verify each against the publisher's page before submission.
- **Build:** `./build.sh` (needs a TeX distribution with `latexmk`), or upload `paper/` to
  Overleaf. CI compiles the paper on every push.
