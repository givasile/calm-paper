# code

Everything that runs, and its results. Run every command from this folder.

## The tables, without fitting anything

```bash
uv sync                                               # the exact environment (uv.lock; Linux x86-64, Python 3.11)
uv run python reports/tables.py                       # Tables 1-2, the interaction and runtime tables
uv run python reports/depth.py --tex results/tables   # gap closed per depth, the depth ladders
uv run python reports/sensitivity.py                  # half and double the selection threshold
uv run python reports/complexity.py                   # interactions, splits and curves per dataset
```

They read the collected results in `results/` and write to `results/tables/`;
the `.tex` files there are the ones the paper includes. `reports/stability.py`
also needs the selection chains each fit writes under `runs/`, so it runs
after `main` and `stability`.

## Running it

```bash
uv run calm-paper run main    # fit every missing cell of an experiment
./run.sh                      # everything, in paper order, detached in tmux
uv run calm-paper status      # what is done, what is left
uv run calm-paper compare     # new results against the frozen ones
uv run calm-paper collect     # write results/<experiment>.csv
```

A *cell* is one fit: (experiment, dataset, arm, fold). A finished cell is a
file under `runs/` and is never fitted again, so a run can be stopped and
resumed at any point. HELOC and MIMIC2 must be obtained first
([`data/README.md`](data/README.md)).

## Experiments

| experiment | serves | cells | fit time, summed |
|---|---|---|---|
| `main` | Tables 1-2, interaction tables, complexity counts | 375 | 4h17m |
| `eb2m` | the GA²M column of Tables 1-2 | 125 | 11h13m |
| `depth` | gap closed per depth | 250 | 7h49m |
| `sensitivity` | half and double the selection threshold | 250 | 8h12m |
| `depth_ladders` | depths 4-6 on Adult, Bank, Bike Sharing, Phoneme | 60 | 4h04m |
| `stability` | bootstrap stability of CALM's regions | 500 | 91h20m |
| `stability_eb2m` | bootstrap stability of the GA²M's pairs | 500 | 314h34m |
| `synthetic` | the explanatory example | 45 | 57m |

## Reproduction

The suite was rerun from scratch on the released `calm-additive==0.1.0` and
`effector==0.6.0`. All 1,920 cells that have a counterpart in `frozen/` (the
runs behind the submitted paper) match within 5e-5
([`results/COMPARE.txt`](results/COMPARE.txt)), and the tables come out
byte-identical to the paper's.

## Layout

| path | what it is |
|---|---|
| `calm_paper/datasets/` | the loaders: 25 benchmark datasets, 3 synthetic ones |
| `calm_paper/protocol.py` | seeding, encoding, splits, metrics |
| `calm_paper/methods.py` | XGBoost, EBM / EB²M, CALM |
| `calm_paper/store.py`, `run.py` | one file per cell; the workers |
| `calm_paper/compare.py` | new results against `frozen/` |
| `experiments/*.yaml` | one experiment each: datasets × arms × resampling |
| `reports/` | results → the paper's tables and figures |
| `results/` | collected results, one CSV per experiment; tables and figures |
| `frozen/` | the fold-level results behind the submitted paper, with checksums |
| `submission/` | the submitted tables, for the columns kept as submitted |
| `data/` | the datasets that are not fetched, and their sources |

Citation and licence: [`../README.md`](../README.md).
