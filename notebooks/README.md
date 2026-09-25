# notebooks/

Thin Colab/Kaggle wrappers around tested entrypoints. Each notebook runs a smoke
version first; all results go to the experiment DB (`results/`) and are merged locally
with `python -m edge_ai_compression.experiment_db.merge`. Keep outputs out of git.

| Notebook | Runs |
|---|---|
| `phase3_ladder.ipynb` | Baseline training (3 seeds) + PTQ/QAT ladder sweeps (`run_phase3`) |
