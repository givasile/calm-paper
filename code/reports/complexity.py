#!/usr/bin/env python3
"""How complex the fitted CALM models are, per dataset.

    python reports/complexity.py [--out DIR]

For every dataset, mean and std over the 5 folds of experiment `main`:

  pairs    conditional interactions, sum over features of |S_i|
  splits   curves added on top of a GAM (one per split)
  curves   regional curves, on the features that were partitioned
  total    all the curves a reader of the model looks at

Writes `complexity.md`.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import datasets, summary  # noqa: E402

COUNTS = {
    "pairs": "n_conditional_interactions",
    "splits": "n_splits",
    "curves": "n_regions",
    "total": "n_curves",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = summary.cells("main")
    rows = rows[rows["arm"] == "calm"]
    lines = []
    for task, names in datasets.BENCHMARK.items():
        part = rows[rows["dataset"].isin(names)]
        g = part.groupby("dataset")[list(COUNTS.values())]
        mean, std, folds = g.mean(), g.std(ddof=1), g.size()
        lines += [f"### {task} ({len(mean)} of {len(names)} datasets)", "",
                  "| dataset | folds | " + " | ".join(COUNTS) + " |", "|---|---|" + "---|" * len(COUNTS)]
        for name in names:
            if name not in mean.index:
                continue
            cells = " | ".join(f"{mean.loc[name, c]:.1f} ± {std.loc[name, c]:.1f}" for c in COUNTS.values())
            lines.append(f"| {name} | {folds[name]} | {cells} |")
        lines.append("| **mean over datasets** | | " + " | ".join(f"**{mean[c].mean():.1f}**" for c in COUNTS.values()) + " |")
        lines.append("")
    text = "\n".join(lines)
    (args.out / "complexity.md").write_text(text)
    print(text)
