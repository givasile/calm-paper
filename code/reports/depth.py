#!/usr/bin/env python3
"""How much of the gap between the GAM and the black box CALM closes, by depth.

    python reports/depth.py [--source new|frozen] [--out DIR]

gap closed = (CALM - GAM) / (XGB - GAM) on the headline metric, per dataset.
A dataset with no gap to close (the GAM is within half an XGB fold-std of the
black box, or ahead of it) is left out of the summary and listed. Fractions
are never capped; the summary is the median and the quartiles over datasets.

Also the depth ladders (depths 1 to 6) of the datasets that have them.
Writes `depth.md`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import summary  # noqa: E402

DROP = {"Electrical"}  # target leak — out of every table and count

DEPTH = {"calm_depth1": 1, "calm": 2, "calm_depth3": 3, "calm_depth4": 4, "calm_depth5": 5, "calm_depth6": 6}


def report(source):
    t = summary.per_dataset(["main", "depth", "depth_ladders"], source)
    t = t[~t["dataset"].isin(DROP)]
    lines = []
    for task in ("regression", "classification"):
        part = t[t["task"] == task]
        metric = part["metric"].iloc[0]
        score = summary.SIGN[metric] * part.pivot(index="dataset", columns="arm", values="score")
        pairs = part.pivot(index="dataset", columns="arm", values="pairs")
        xgb_std = part[part["arm"] == "xgb"].set_index("dataset")["std"]
        gap = score["xgb"] - score["ebm"]
        no_gap = gap[(gap <= 0) | (gap < 0.5 * xgb_std)].index
        closed = pd.DataFrame(
            {a: (score[a] - score["ebm"]) / gap for a in DEPTH if a in score}
        ).drop(index=no_gap)

        lines += [f"## {task} — gap closed, {len(closed)} of {len(score)} datasets", "",
                  "| depth | datasets | median | quartiles | half closed | median pairs, all datasets |", "|---|---|---|---|---|---|"]
        for arm in ("calm_depth1", "calm", "calm_depth3"):
            if arm not in closed or closed[arm].isna().all():
                continue
            c = closed[arm].dropna()
            q1, med, q3 = c.quantile([0.25, 0.5, 0.75])
            lines.append(
                f"| {DEPTH[arm]} | {len(c)} | {med:.0%} | {q1:.0%} – {q3:.0%} | "
                f"{int((c >= 0.5).sum())} | {pairs[arm].median():.1f} |"
            )
        lines += ["", "no gap to close: " + (", ".join(no_gap) or "none"), ""]

        ladder = [d for d in score.index if "calm_depth4" in score and pd.notna(score.loc[d, "calm_depth4"])]
        for d in ladder:
            lines += [f"### {d} — depth ladder ({metric})", "",
                      "| depth | " + metric + " | pairs | gap closed |", "|---|---|---|---|"]
            row = part[part["dataset"] == d].set_index("arm")
            for label, arm in [("GAM", "ebm")] + [(str(k), a) for a, k in DEPTH.items()] + [("XGB", "xgb")]:
                if arm not in row.index:
                    continue
                r = row.loc[arm]
                # with no gap to close the fraction is a ratio of two noises
                frac = "–" if d in no_gap else f"{(score.loc[d, arm] - score.loc[d, 'ebm']) / gap[d]:.0%}"
                p = "" if pd.isna(r["pairs"]) else f"{r['pairs']:.1f}"
                lines.append(f"| {label} | {r['score']:.4g} ± {r['std']:.2g} | {p} | {frac} |")
            lines.append("")
    return "\n".join(lines)


LADDERS = {"BikeSharing": "Bike Sharing", "Adult": "Adult", "Bank": "Bank", "PMLB_PHONEME": "Phoneme"}


def ladder_tex(source, out):
    """Rows of the appendix depth-ladder table: score and pairs per depth, four datasets."""
    t = summary.per_dataset(["main", "depth", "depth_ladders"], source)
    t = t[~t["dataset"].isin(DROP)].set_index(["dataset", "arm"])
    rows = []
    for label, arm in [("GAM", "ebm")] + [(str(k), a) for a, k in DEPTH.items()] + [("XGB", "xgb")]:
        cells = [label]
        for d in LADDERS:
            r = t.loc[(d, arm)]
            dec = 2 if r["metric"] == "rmse" else 4
            cells.append(rf"${r['score']:.{dec}f} \text{{\scriptsize$\pm${r['std']:.{dec}f}}}$")
            cells.append("" if pd.isna(r["pairs"]) or arm in ("ebm", "xgb") else f"{r['pairs']:.1f}")
        rows.append(" & ".join(cells) + r" \\")
        if label in ("GAM", "6"):
            rows.append(r"\midrule")
    (out / "tab_depth_ladder_rows.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["new", "frozen"], default="new")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--tex", type=Path, help="also write the appendix ladder rows here")
    args = ap.parse_args()
    if args.tex:
        ladder_tex(args.source, args.tex)
    args.out.mkdir(parents=True, exist_ok=True)
    text = report(args.source)
    (args.out / "depth.md").write_text(text + "\n")
    print(text)
