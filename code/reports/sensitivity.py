#!/usr/bin/env python3
"""What halving or doubling the selection threshold does.

    python reports/sensitivity.py [--source new|frozen] [--out DIR]

Each variant against the default (min_r2_gain = 0.01), per dataset: a win or
a loss when the means differ by more than the larger of the two fold stds, a
draw otherwise; and the change in the number of conditional interactions.
Writes `sensitivity.md`.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import summary  # noqa: E402

DROP = {"Electrical"}  # target leak — out of every table and count

VARIANTS = {"calm_gain_half": "half (0.005)", "calm_gain_double": "double (0.02)"}


def report(source):
    t = summary.per_dataset(["main", "sensitivity"], source)
    t = t[~t["dataset"].isin(DROP)]
    base = t[t["arm"] == "calm"].set_index("dataset")
    lines = ["| threshold | task | datasets | W / D / L | mean change in pairs | losses |", "|---|---|---|---|---|---|"]
    for arm, label in VARIANTS.items():
        v = t[t["arm"] == arm].set_index("dataset")
        total = [0, 0, 0]
        for task in ("regression", "classification", "both"):
            part = v if task == "both" else v[v["task"] == task]
            if part.empty:
                continue
            b = base.loc[part.index]
            sign = part["metric"].map(summary.SIGN)
            good = sign * (part["score"] - b["score"])
            spread = part["std"].combine(b["std"], max).clip(lower=1e-12)
            w, l = int((good > spread).sum()), int((good < -spread).sum())
            d = len(part) - w - l
            lost = ", ".join(part.index[good < -spread])
            lines.append(
                f"| {label} | {task} | {len(part)} | {w} / {d} / {l} | "
                f"{(part['pairs'] - b['pairs']).mean():+.1f} | {lost} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["new", "frozen"], default="new")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    text = report(args.source)
    (args.out / "sensitivity.md").write_text(text + "\n")
    print(text)
