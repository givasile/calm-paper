"""New results against the frozen ones, cell by cell.

`frozen/results/` holds the fold-level numbers behind the submitted paper and
the rebuttal. An experiment names the frozen file(s) it reruns and, per arm,
the method name used there; a cell *matches* when every metric (and, for
CALM, the number of conditional interactions) agrees within `TOLERANCE`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calm_paper import protocol, store
from calm_paper.store import ROOT

FROZEN = ROOT / "frozen" / "results"
TOLERANCE = 5e-5  # four digits
MODEL = {"regression": "Regressor", "classification": "Classifier"}


def frozen_rows(spec: dict) -> pd.DataFrame:
    """The frozen fold rows an experiment reruns, labelled with its arm names."""
    patterns = spec.get("frozen")
    if not patterns:
        return pd.DataFrame()
    patterns = [patterns] if isinstance(patterns, str) else patterns
    frames = []
    for task in protocol.METRICS:
        files = sorted(f for p in patterns for f in FROZEN.glob(p.format(task=task)))
        if not files:
            continue
        old = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        old = old[old["task"] == task]
        for arm, a in spec["arms"].items():
            if "frozen" not in a:
                continue
            part = old[old["method"] == a["frozen"].format(Model=MODEL[task])].copy()
            part["arm"] = arm
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["dataset", "arm", "fold"], keep="last")


def compare(spec: dict) -> pd.DataFrame:
    """One row per cell present on both sides: the largest difference found."""
    new, old = store.rows(spec["name"]), frozen_rows(spec)
    if new.empty or old.empty:
        return pd.DataFrame()
    merged = new.merge(old, on=["dataset", "arm", "fold"], suffixes=("", "_frozen"))
    out = []
    for _, r in merged.iterrows():
        diffs = {m: abs(r[m] - r[m + "_frozen"]) for m in protocol.METRICS[r["task"]]}
        if r["method"] == "calm" and not pd.isna(r.get("num_interactions")):
            diffs["interactions"] = abs(r["n_conditional_interactions"] - r["num_interactions"])
        head = protocol.HEADLINE[r["task"]]
        out.append(
            {
                "dataset": r["dataset"], "arm": r["arm"], "fold": int(r["fold"]),
                "metric": head, "new": r[head], "frozen": r[head + "_frozen"],
                "max_diff": float(np.nanmax(list(diffs.values()))),
                "seconds": r["runtime_sec"], "seconds_frozen": r["runtime_sec_frozen"],
            }
        )
    out = pd.DataFrame(out)
    out["match"] = out["max_diff"] <= TOLERANCE
    return out
