"""Fold results as the reports read them: from this suite, or the frozen runs."""

from __future__ import annotations

import pandas as pd

from calm_paper import compare, protocol, run, store

SIGN = {"rmse": -1, "accuracy": +1}  # turns the headline metric into higher-is-better


def cells(experiment: str) -> pd.DataFrame:
    """Every finished cell of an experiment: from `runs/`, or, in a clone
    without runs, from the collected `results/<experiment>.csv`."""
    out = store.rows(experiment)
    path = store.ROOT / "results" / f"{experiment}.csv"
    if out.empty and path.exists():
        out = pd.read_csv(path)
    return out


def folds(experiment: str, source: str = "new") -> tuple[pd.DataFrame, int]:
    """Fold rows of an experiment and the number of cells each arm has when complete.

    `source="frozen"` returns the frozen rows the experiment reruns, under the
    same column names, so a report can be checked against the old numbers.
    """
    spec = run.load(experiment)
    if source == "frozen":
        rows = compare.frozen_rows(spec).rename(columns={"num_interactions": "n_conditional_interactions"})
    else:
        rows = cells(experiment)
    return rows, len(spec["datasets"]) * int(spec["resampling"]["n"])


def per_dataset(experiments: list[str], source: str = "new") -> pd.DataFrame:
    """One row per (dataset, arm) with all its folds present.

    `score` is the headline metric (accuracy, or RMSE in the target's units),
    `std` its population std over the folds, `pairs` the mean number of
    conditional interactions.
    """
    out = []
    for experiment in experiments:
        rows, _ = folds(experiment, source)
        if rows.empty:
            continue
        n_folds = int(run.load(experiment)["resampling"]["n"])
        for (dataset, arm), g in rows.groupby(["dataset", "arm"]):
            if len(g) != n_folds:
                continue
            task = g["task"].iloc[0]
            metric = protocol.HEADLINE[task]
            out.append(
                {
                    "dataset": dataset, "arm": arm, "task": task, "metric": metric,
                    "score": g[metric].mean(), "std": g[metric].std(ddof=0),
                    "pairs": g["n_conditional_interactions"].mean()
                    if "n_conditional_interactions" in g else float("nan"),
                }
            )
    return pd.DataFrame(out)
