# Frozen results

The fold-level results behind the submitted paper and the rebuttal, kept as
they were produced. `calm-paper compare` checks every new cell against them.

| path | what it is |
|---|---|
| `results/*_folds.csv` | one row per (dataset, method, fold) |
| `results/*.csv` | the same, averaged over folds |
| `results/ebm_terms/` | the pairs each EB²M selected, per fold |
| `MANIFEST.sha256` | checksums of everything under `results/` (`sha256sum -c MANIFEST.sha256`) |
| `environment.txt` | the Python environment of those runs |

These runs were made with an earlier, unreleased development version of the
engine (the `effector-calm` line in `environment.txt`). Rerun from scratch on
the released `calm-additive==0.1.0` and `effector==0.6.0`, every cell that has
a counterpart here matches within 5e-5 (`results/COMPARE.txt`).
